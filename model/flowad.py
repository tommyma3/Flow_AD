import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model

from env import map_dark_states


class FlowAD(torch.nn.Module):

    def __init__(self, config):
        super(FlowAD, self).__init__()

        self.config = config
        self.device = config['device']
        self.n_transit = config['n_transit']
        self.context_len = self.n_transit - 1
        self.max_seq_length = 3 * self.context_len + 3
        self.mixed_precision = config['mixed_precision']
        self.grid_size = config['grid_size']
        self.num_actions = config['num_actions']

        tf_n_embd = config['tf_n_embd']
        tf_n_head = config.get('tf_n_head', 4)
        tf_n_layer = config.get('tf_n_layer', 4)
        tf_n_inner = config.get('tf_n_inner', config.get('tf_dim_feedforward', tf_n_embd * 4))
        tf_dropout = config.get('tf_dropout', 0.1)
        tf_attn_dropout = config.get('tf_attn_dropout', 0.1)

        gpt2_cfg = GPT2Config(
            n_positions=self.max_seq_length,
            n_embd=tf_n_embd,
            n_layer=tf_n_layer,
            n_head=tf_n_head,
            n_inner=tf_n_inner,
            resid_pdrop=tf_dropout,
            embd_pdrop=tf_dropout,
            attn_pdrop=tf_attn_dropout,
            use_cache=False,
        )
        gpt2_cfg._attn_implementation = 'eager'
        self.transformer_model = GPT2Model(gpt2_cfg)

        self.embed_state = nn.Embedding(config['grid_size'] * config['grid_size'], tf_n_embd)
        self.embed_action = nn.Embedding(config['num_actions'], tf_n_embd)
        self.embed_reward = nn.Linear(1, tf_n_embd)

        self.embed_flow_action = nn.Linear(config['num_actions'], tf_n_embd)
        self.embed_time = nn.Sequential(
            nn.Linear(1, tf_n_embd),
            nn.SiLU(),
            nn.Linear(tf_n_embd, tf_n_embd),
        )

        self.pred_velocity = nn.Sequential(
            nn.Linear(tf_n_embd, tf_n_embd),
            nn.SiLU(),
            nn.Linear(tf_n_embd, config['num_actions']),
        )
        self.pred_policy = nn.Linear(tf_n_embd, config['num_actions'])

        self.flow_train_steps = config.get('flow_train_steps', 6)
        self.flow_eval_steps = config.get('flow_eval_steps', 10)
        self.flow_velocity_scale = float(config.get('flow_velocity_scale', 1.0))
        self.flow_action_noise_std = float(config.get('flow_action_noise_std', 0.01))
        self.flow_policy_loss_weight = float(config.get('flow_policy_loss_weight', 1.0))
        self.flow_prior_init_mix = float(config.get('flow_prior_init_mix', 0.1))

    def transformer(self, x, return_attentions=False):
        output = self.transformer_model(
            inputs_embeds=x,
            output_attentions=return_attentions,
            return_dict=True,
            use_cache=False,
        )
        attentions = list(output.attentions) if return_attentions else None
        return output.last_hidden_state, attentions

    def _normalize_policy(self, action_prob):
        action_prob = torch.clamp(action_prob, min=1e-6)
        return action_prob / action_prob.sum(dim=-1, keepdim=True)

    def _sample_base_policy(self, batch_size, device):
        base_policy = torch.rand((batch_size, self.num_actions), device=device, dtype=torch.float)
        return self._normalize_policy(base_policy)

    def _build_token_sequence(self, states, actions, rewards, query_states, action_t, t):
        state_ids = map_dark_states(states.to(torch.long), self.grid_size)
        query_state_ids = map_dark_states(query_states.to(torch.long), self.grid_size)

        state_tokens = self.embed_state(state_ids)
        action_tokens = self.embed_action(actions.to(torch.long))
        reward_tokens = self.embed_reward(rewards.unsqueeze(-1).to(torch.float))

        batch_size = state_tokens.size(0)
        context_tokens = torch.stack([state_tokens, action_tokens, reward_tokens], dim=2)
        context_tokens = context_tokens.reshape(batch_size, -1, context_tokens.size(-1))

        query_token = self.embed_state(query_state_ids).unsqueeze(1)
        flow_action_token = self.embed_flow_action(action_t.to(torch.float)).unsqueeze(1)
        time_token = self.embed_time(t.to(torch.float)).unsqueeze(1)

        return torch.cat([context_tokens, query_token, flow_action_token, time_token], dim=1)

    def _forward_flow(self, states, actions, rewards, query_states, action_t, t, return_attentions=False):
        transformer_input = self._build_token_sequence(
            states=states,
            actions=actions,
            rewards=rewards,
            query_states=query_states,
            action_t=action_t,
            t=t,
        )
        transformer_output, attentions = self.transformer(transformer_input, return_attentions=return_attentions)

        velocity = self.pred_velocity(transformer_output[:, -1])
        policy_logits = self.pred_policy(transformer_output[:, -3])
        return velocity, policy_logits, attentions

    def _get_optimal_policy(self, query_states, query_goals):
        query_states = query_states.to(torch.long)
        query_goals = query_goals.to(torch.long)

        dx = query_goals[:, 0] - query_states[:, 0]
        dy = query_goals[:, 1] - query_states[:, 1]

        optimal_policy = torch.zeros((query_states.size(0), self.num_actions), device=query_states.device, dtype=torch.float)
        optimal_policy[:, 0] = (dx > 0).to(torch.float)
        optimal_policy[:, 1] = (dx < 0).to(torch.float)
        optimal_policy[:, 2] = (dy > 0).to(torch.float)
        optimal_policy[:, 3] = (dy < 0).to(torch.float)

        no_move = (dx == 0) & (dy == 0)
        optimal_policy[:, 4] = no_move.to(torch.float)

        return self._normalize_policy(optimal_policy)

    def _predict_prior_policy(self, states, actions, rewards, query_states, return_attentions=False):
        batch_size = query_states.size(0)
        init_action = torch.full(
            (batch_size, self.num_actions),
            fill_value=1.0 / self.num_actions,
            device=query_states.device,
            dtype=torch.float,
        )
        t0 = torch.zeros((batch_size, 1), device=query_states.device, dtype=torch.float)
        _, policy_logits, attentions = self._forward_flow(
            states=states,
            actions=actions,
            rewards=rewards,
            query_states=query_states,
            action_t=init_action,
            t=t0,
            return_attentions=return_attentions,
        )
        return F.softmax(policy_logits, dim=-1), policy_logits, attentions

    def _integrate_flow(self, states, actions, rewards, query_states, initial_policy, steps, return_attentions=False):
        action_prob = self._normalize_policy(initial_policy)
        batch_size = query_states.size(0)
        last_attentions = None

        if steps <= 0:
            return action_prob, last_attentions

        dt = 1.0 / steps

        for k in range(steps):
            t = torch.full(
                (batch_size, 1),
                fill_value=(k + 0.5) / steps,
                device=query_states.device,
                dtype=torch.float,
            )
            velocity, _, attentions = self._forward_flow(
                states=states,
                actions=actions,
                rewards=rewards,
                query_states=query_states,
                action_t=action_prob,
                t=t,
                return_attentions=return_attentions,
            )
            velocity = velocity / max(self.flow_velocity_scale, 1e-6)
            action_prob = self._normalize_policy(action_prob + dt * velocity)
            last_attentions = attentions

        return action_prob, last_attentions

    def forward(self, x):
        query_states = x['query_states'].to(self.device).to(torch.long)
        query_goals = x['query_goals'].to(self.device).to(torch.long)
        states = x['states'].to(self.device)
        actions = x['actions'].to(self.device)
        rewards = x['rewards'].to(self.device)

        target_policy = self._get_optimal_policy(query_states, query_goals)

        batch_size = query_states.size(0)
        base_policy = self._sample_base_policy(batch_size=batch_size, device=self.device)
        t = torch.rand((batch_size, 1), device=self.device, dtype=torch.float)

        action_t = (1.0 - t) * base_policy + t * target_policy
        if self.flow_action_noise_std > 0.0:
            action_t = action_t + self.flow_action_noise_std * torch.randn_like(action_t)
        action_t = self._normalize_policy(action_t)

        target_velocity = target_policy - base_policy

        velocity_pred, _, attentions = self._forward_flow(
            states=states,
            actions=actions,
            rewards=rewards,
            query_states=query_states,
            action_t=action_t,
            t=t,
            return_attentions=False,
        )

        prior_policy, prior_logits, _ = self._predict_prior_policy(
            states=states,
            actions=actions,
            rewards=rewards,
            query_states=query_states,
            return_attentions=False,
        )

        loss_flow = F.mse_loss(velocity_pred, target_velocity)
        prior_log_probs = F.log_softmax(prior_logits, dim=-1)
        loss_policy = -(target_policy * prior_log_probs).sum(dim=-1).mean()
        loss = loss_flow + self.flow_policy_loss_weight * loss_policy

        with torch.no_grad():
            refined_policy, _ = self._integrate_flow(
                states=states,
                actions=actions,
                rewards=rewards,
                query_states=query_states,
                initial_policy=prior_policy,
                steps=self.flow_train_steps,
                return_attentions=False,
            )
            pred_actions = refined_policy.argmax(dim=-1)
            acc_optimal = target_policy.gather(1, pred_actions.unsqueeze(1)).squeeze(1).mean()

        return {
            'loss': loss,
            'loss_flow': loss_flow,
            'loss_policy': loss_policy,
            'acc_action': acc_optimal,
            'acc_action_per_state': acc_optimal,
            'attentions': attentions,
        }

    def evaluate_in_context(self, vec_env, eval_timesteps, beam_k=0, sample=True, return_attentions=False):
        outputs = {}
        outputs['reward_episode'] = []

        reward_episode = np.zeros(vec_env.num_envs)

        query_states = torch.tensor(
            vec_env.reset(),
            device=self.device,
            requires_grad=False,
            dtype=torch.long,
        )
        states_hist = torch.empty(
            (vec_env.num_envs, 0, query_states.size(-1)),
            device=self.device,
            dtype=torch.long,
        )
        actions_hist = torch.empty(
            (vec_env.num_envs, 0),
            device=self.device,
            dtype=torch.long,
        )
        rewards_hist = torch.empty(
            (vec_env.num_envs, 0),
            device=self.device,
            dtype=torch.float,
        )

        if return_attentions:
            per_step_attentions = []
            dones_history = []

        for _ in range(eval_timesteps):
            query_states_prev = query_states.clone().detach()

            prior_policy, _, _ = self._predict_prior_policy(
                states=states_hist,
                actions=actions_hist,
                rewards=rewards_hist,
                query_states=query_states,
                return_attentions=False,
            )
            uniform_policy = torch.full_like(prior_policy, fill_value=1.0 / self.num_actions)
            init_policy = self._normalize_policy(
                (1.0 - self.flow_prior_init_mix) * prior_policy + self.flow_prior_init_mix * uniform_policy
            )

            action_policy, attentions = self._integrate_flow(
                states=states_hist,
                actions=actions_hist,
                rewards=rewards_hist,
                query_states=query_states,
                initial_policy=init_policy,
                steps=self.flow_eval_steps,
                return_attentions=return_attentions,
            )

            if return_attentions:
                per_step_attentions.append([a.detach().cpu().clone() for a in attentions])

            if sample:
                actions = torch.multinomial(action_policy, num_samples=1).squeeze(1)
            else:
                actions = action_policy.argmax(dim=-1)

            query_states_np, rewards_np, dones, infos = vec_env.step(actions.cpu().numpy())

            if return_attentions:
                dones_history.append(dones.copy())

            reward_episode += rewards_np
            rewards = torch.tensor(
                rewards_np,
                device=self.device,
                requires_grad=False,
                dtype=torch.float,
            )

            if dones[0]:
                outputs['reward_episode'].append(reward_episode)
                reward_episode = np.zeros(vec_env.num_envs)

            query_states = torch.tensor(
                query_states_np,
                device=self.device,
                requires_grad=False,
                dtype=torch.long,
            )

            states_hist = torch.cat([states_hist, query_states_prev.unsqueeze(1)], dim=1)
            actions_hist = torch.cat([actions_hist, actions.unsqueeze(1)], dim=1)
            rewards_hist = torch.cat([rewards_hist, rewards.unsqueeze(1)], dim=1)

            states_hist = states_hist[:, -self.context_len:]
            actions_hist = actions_hist[:, -self.context_len:]
            rewards_hist = rewards_hist[:, -self.context_len:]

        outputs['reward_episode'] = np.stack(outputs['reward_episode'], axis=1)

        if return_attentions:
            outputs['attentions'] = per_step_attentions
            outputs['dones_history'] = dones_history

        return outputs

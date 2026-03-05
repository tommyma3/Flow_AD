import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model

from env import map_dark_states

class AD(torch.nn.Module):

    def __init__(self, config):
        super(AD, self).__init__()

        self.config = config
        self.device = config['device']
        self.n_transit = config['n_transit']
        self.context_len = self.n_transit - 1
        self.max_seq_length = 3 * self.context_len + 1
        self.mixed_precision = config['mixed_precision']
        self.grid_size = config['grid_size']

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
        self.pred_action = nn.Linear(tf_n_embd, config['num_actions'])

        self.loss_fn = nn.CrossEntropyLoss(reduction='mean', label_smoothing=config['label_smoothing'])

    def transformer(self, x, max_seq_length=None, dtype=None, return_attentions=False):
        """
        Returns:
            out: (batch, seq, emb)
            attentions: list length=num_layers; each tensor (B, H, L, L)
        """
        output = self.transformer_model(
            inputs_embeds=x,
            output_attentions=return_attentions,
            return_dict=True,
            use_cache=False,
        )
        attentions = list(output.attentions) if return_attentions else None
        return output.last_hidden_state, attentions

    def _build_token_sequence(self, states, actions, rewards, query_states):
        state_ids = map_dark_states(states.to(torch.long), self.grid_size)
        query_state_ids = map_dark_states(query_states.to(torch.long), self.grid_size)

        state_tokens = self.embed_state(state_ids)
        action_tokens = self.embed_action(actions.to(torch.long))
        reward_tokens = self.embed_reward(rewards.unsqueeze(-1).to(torch.float))

        batch_size = state_tokens.size(0)
        context_tokens = torch.stack([state_tokens, action_tokens, reward_tokens], dim=2)
        context_tokens = context_tokens.reshape(batch_size, -1, context_tokens.size(-1))

        query_token = self.embed_state(query_state_ids).unsqueeze(1)
        return torch.cat([context_tokens, query_token], dim=1)

    def _predict_actions_from_tokens(self, transformer_output):
        # State tokens live at positions 0, 3, 6, ... and include the final query state.
        return self.pred_action(transformer_output[:, 0::3])

    def forward(self, x):
        query_states = x['query_states'].to(self.device)  # (batch_size, dim_state)
        target_actions = x['target_actions'].to(self.device)  # (batch_size,)
        states = x['states'].to(self.device)  # (batch_size, num_transit - 1, dim_state)
        actions = x['actions'].to(self.device)  # (batch_size, num_transit - 1)
        rewards = x['rewards'].to(self.device)  # (batch_size, num_transit - 1)

        transformer_input = self._build_token_sequence(
            states=states,
            actions=actions,
            rewards=rewards,
            query_states=query_states,
        )

        transformer_output, attentions = self.transformer(transformer_input,
                                              max_seq_length=self.max_seq_length,
                                              dtype=self.mixed_precision,
                                              return_attentions=False)

        result = {}
        logits_actions = self._predict_actions_from_tokens(transformer_output)
        target_actions_seq = torch.cat([actions, target_actions.unsqueeze(1)], dim=1)

        loss_full_action = self.loss_fn(
            logits_actions.reshape(-1, logits_actions.size(-1)),
            target_actions_seq.reshape(-1),
        )
        acc_full_action = (
            logits_actions.argmax(dim=-1) == target_actions_seq
        ).float().mean()

        result['loss'] = loss_full_action
        result['loss_action'] = loss_full_action
        result['acc_action'] = acc_full_action
        result['attentions'] = attentions

        return result

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

        transformer_input = self._build_token_sequence(
            states=states_hist,
            actions=actions_hist,
            rewards=rewards_hist,
            query_states=query_states,
        )

        if return_attentions:
            per_step_attentions = []
            dones_history = []

        for step in range(eval_timesteps):
            query_states_prev = query_states.clone().detach()

            output, attentions = self.transformer(transformer_input,
                                        max_seq_length=self.max_seq_length,
                                        dtype='fp32',
                                        return_attentions=return_attentions)

            if return_attentions:
                per_step_attentions.append([a.detach().cpu().clone() for a in attentions])

            logits = self._predict_actions_from_tokens(output)[:, -1]

            if sample:
                log_probs = F.log_softmax(logits, dim=-1)
                actions = torch.multinomial(log_probs.exp(), num_samples=1)
                actions = actions.squeeze(1)
            else:
                actions = logits.argmax(dim=-1)

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

            transformer_input = self._build_token_sequence(
                states=states_hist,
                actions=actions_hist,
                rewards=rewards_hist,
                query_states=query_states,
            )

        outputs['reward_episode'] = np.stack(outputs['reward_episode'], axis=1)

        if return_attentions:
            outputs['attentions'] = per_step_attentions
            outputs['dones_history'] = dones_history

        return outputs

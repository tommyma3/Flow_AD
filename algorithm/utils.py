import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class HistoryLoggerCallback(BaseCallback):
    def __init__(self, env_name, env_idx, history=None):
        super(HistoryLoggerCallback, self).__init__()
        self.env_name = env_name
        self.env_idx = env_idx

        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

        self.history = history

        self.episode_rewards = []
        self.episode_success = []

    def _on_step(self) -> bool:
        # Capture state, action, and reward at each step
        self.states.append(self.locals["obs_tensor"].cpu().numpy())
        self.next_states.append(self.locals["new_obs"])
        self.actions.append(self.locals["actions"])

        self.rewards.append(self.locals["rewards"].copy())
        self.dones.append(self.locals["dones"])

        self.episode_rewards.append(self.locals['rewards'])
        
        if self.locals['dones'][0]:
            mean_reward = np.mean(np.mean(self.episode_rewards, axis=0))
            self.logger.record('rollout/mean_reward', mean_reward)
            self.episode_rewards = []
                        
        return True

    def _on_training_end(self):
        self.history[self.env_idx] = {
            'states': np.array(self.states, dtype=np.int32),
            'actions': np.array(self.actions, dtype=np.int32),
            'rewards': np.array(self.rewards, dtype=np.int32),
            'next_states': np.array(self.next_states, dtype=np.int32),
            'dones': np.array(self.dones, dtype=np.bool_)
        }
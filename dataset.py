from torch.utils.data import Dataset
import numpy as np
from utils import get_traj_file_name
import h5py
import random


class ADDataset(Dataset):
    def __init__(self, config, traj_dir, mode='train', n_stream=None, source_timesteps=None):
        self.config = config
        self.env = config['env']
        self.n_transit = config['n_transit']
        self.dynamics = config['dynamics']

        if self.env == 'darkroom':
            n_total_envs = config['grid_size'] ** 2
        else:
            raise ValueError('Invalid env')

        total_env_idx = list(range(n_total_envs))
        random.seed(config['env_split_seed'])
        random.shuffle(total_env_idx)

        n_train_envs = round(n_total_envs * config['train_env_ratio'])

        if mode == 'train':
            env_idx = total_env_idx[:n_train_envs]
        elif mode == 'test':
            env_idx = total_env_idx[n_train_envs:]
        elif mode == 'all':
            env_idx = total_env_idx
        else:
            raise ValueError('Invalid mode')

        states = []
        actions = []
        rewards = []
        next_states = []
        goals = []

        with h5py.File(f'{traj_dir}/{get_traj_file_name(config)}.hdf5', 'r') as f:
            for i in env_idx:
                states_i = f[f'{i}']['states'][()].transpose(1, 0, 2)[:n_stream, :source_timesteps]
                actions_i = f[f'{i}']['actions'][()].transpose(1, 0)[:n_stream, :source_timesteps]
                rewards_i = f[f'{i}']['rewards'][()].transpose(1, 0)[:n_stream, :source_timesteps]
                next_states_i = f[f'{i}']['next_states'][()].transpose(1, 0, 2)[:n_stream, :source_timesteps]

                states.append(states_i)
                actions.append(actions_i)
                rewards.append(rewards_i)
                next_states.append(next_states_i)

                if self.env == 'darkroom':
                    goal = np.array([i // self.config['grid_size'], i % self.config['grid_size']], dtype=np.int32)
                else:
                    raise ValueError('Invalid env')
                goals.append(np.repeat(goal[None, :], repeats=states_i.shape[0], axis=0))

        self.states = np.concatenate(states, axis=0)
        self.actions = np.concatenate(actions, axis=0)
        self.rewards = np.concatenate(rewards, axis=0)
        self.next_states = np.concatenate(next_states, axis=0)
        self.goals = np.concatenate(goals, axis=0)

    def __len__(self):
        return (len(self.states[0]) - self.n_transit + 1) * len(self.states)

    def __getitem__(self, i):
        history_idx = i // (len(self.states[0]) - self.n_transit + 1)
        transition_idx = i % (len(self.states[0]) - self.n_transit + 1)

        traj = {
            'query_states': self.states[history_idx, transition_idx + self.n_transit - 1],
            'query_goals': self.goals[history_idx],
            'target_actions': self.actions[history_idx, transition_idx + self.n_transit - 1],
            'states': self.states[history_idx, transition_idx:transition_idx + self.n_transit - 1],
            'actions': self.actions[history_idx, transition_idx:transition_idx + self.n_transit - 1],
            'rewards': self.rewards[history_idx, transition_idx:transition_idx + self.n_transit - 1],
            'next_states': self.next_states[history_idx, transition_idx:transition_idx + self.n_transit - 1],
        }

        if self.dynamics:
            traj.update({
                'target_next_states': self.next_states[history_idx, transition_idx + self.n_transit - 1],
                'target_rewards': self.rewards[history_idx, transition_idx + self.n_transit - 1],
            })

        return traj

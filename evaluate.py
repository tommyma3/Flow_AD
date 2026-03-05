from datetime import datetime
from glob import glob
import argparse

import os
import torch
import os.path as osp

from env import SAMPLE_ENVIRONMENT, make_env
from model import MODEL
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 0


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate AD/FlowAD checkpoint.')
    parser.add_argument('--ckpt-dir', type=str, required=True)
    parser.add_argument('--eval-episodes', type=int, default=500)
    parser.add_argument('--sample', action='store_true', help='Sample actions instead of greedy decode.')
    parser.add_argument('--output', type=str, default='eval_result.npy')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    ckpt_paths = sorted(glob(osp.join(args.ckpt_dir, 'ckpt-*.pt')))

    if len(ckpt_paths) > 0:
        ckpt_path = ckpt_paths[-1]
        ckpt = torch.load(ckpt_path, map_location=device)
        print(f'Checkpoint loaded from {ckpt_path}')
        config = ckpt['config']
        config['device'] = str(device)
    else:
        raise ValueError('No checkpoint found.')

    model_name = config['model']
    model = MODEL[model_name](config).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    env_name = config['env']
    _, test_env_args = SAMPLE_ENVIRONMENT[env_name](config)

    print('Evaluation goals: ', test_env_args)

    if env_name == 'darkroom':
        envs = SubprocVecEnv([make_env(config, goal=arg) for arg in test_env_args])
    else:
        raise NotImplementedError(f'Environment not supported: {env_name}')

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    start_time = datetime.now()
    print(f'Starting at {start_time}')

    eval_timesteps = config['horizon'] * args.eval_episodes

    with torch.no_grad():
        test_rewards = model.evaluate_in_context(
            vec_env=envs,
            eval_timesteps=eval_timesteps,
            sample=args.sample,
        )['reward_episode']
        result_path = osp.join(args.ckpt_dir, args.output)

    end_time = datetime.now()
    print()
    print(f'Ended at {end_time}')
    print(f'Elpased time: {end_time - start_time}')

    envs.close()

    with open(result_path, 'wb') as f:
        np.save(f, test_rewards)

    print('Saved rewards to:', result_path)
    print('Mean reward per environment:', test_rewards.mean(axis=1))
    print('Overall mean reward: ', test_rewards.mean())
    print('Std deviation: ', test_rewards.std())

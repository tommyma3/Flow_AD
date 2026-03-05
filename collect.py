import os
from datetime import datetime
import argparse
import multiprocessing

from env import SAMPLE_ENVIRONMENT, make_env
from algorithm import ALGORITHM, HistoryLoggerCallback
import h5py

from stable_baselines3.common.vec_env import DummyVecEnv

from utils import get_config, get_traj_file_name


def parse_args():
    parser = argparse.ArgumentParser(description='Collect source trajectories for AD/FlowAD.')
    parser.add_argument('--env-config', type=str, default='config/env/darkroom.yaml')
    parser.add_argument('--alg-config', type=str, default='config/algorithm/ppo_darkroom.yaml')
    parser.add_argument('--traj-dir', type=str, default='datasets')
    parser.add_argument('--total-source-timesteps', type=int, default=None)
    parser.add_argument('--n-stream', type=int, default=None)
    parser.add_argument('--n-process', type=int, default=None)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def worker(arg, config, traj_dir, env_idx, history, file_name):
    if config['env'] == 'darkroom':
        env = DummyVecEnv([make_env(config, goal=arg)] * config['n_stream'])
    else:
        raise ValueError('Invalid environment')

    alg_name = config['alg']
    seed = config['alg_seed'] + env_idx

    local_config = dict(config)
    local_config['device'] = 'cpu'

    alg = ALGORITHM[alg_name](local_config, env, seed, traj_dir)
    callback = HistoryLoggerCallback(config['env'], env_idx, history)
    log_name = f'{file_name}_{env_idx}'

    alg.learn(
        total_timesteps=config['total_source_timesteps'],
        callback=callback,
        log_interval=1,
        tb_log_name=log_name,
        reset_num_timesteps=True,
        progress_bar=True,
    )
    env.close()


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    args = parse_args()

    config = get_config(args.env_config)
    config.update(get_config(args.alg_config))

    if args.total_source_timesteps is not None:
        config['total_source_timesteps'] = args.total_source_timesteps
    if args.n_stream is not None:
        config['n_stream'] = args.n_stream
    if args.n_process is not None:
        config['n_process'] = args.n_process

    os.makedirs(args.traj_dir, exist_ok=True)

    train_args, test_args = SAMPLE_ENVIRONMENT[config['env']](config, shuffle=False)
    total_args = train_args + test_args
    n_envs = len(total_args)

    file_name = get_traj_file_name(config)
    file_path = os.path.join(args.traj_dir, f'{file_name}.hdf5')

    if os.path.exists(file_path) and not args.overwrite:
        raise FileExistsError(f'{file_path} already exists. Use --overwrite to regenerate it.')

    start_time = datetime.now()
    print(f'Training started at {start_time}')

    with multiprocessing.Manager() as manager:
        history = manager.dict()

        with multiprocessing.Pool(processes=config['n_process']) as pool:
            pool.starmap(worker, [(total_args[i], config, args.traj_dir, i, history, file_name) for i in range(n_envs)])

        file_mode = 'w' if args.overwrite else 'w-'
        with h5py.File(file_path, file_mode) as f:
            for i in range(n_envs):
                env_group = f.create_group(f'{i}')
                for key, value in history[i].items():
                    env_group.create_dataset(key, data=value)

    end_time = datetime.now()
    print()
    print(f'Training ended at {end_time}')
    print(f'Elapsed time: {end_time - start_time}')

import yaml
import torch
from torch.utils.data import DataLoader
import numpy as np

from env import map_dark_states
from functools import partial
import matplotlib.pyplot as plt


def get_config(config_path):
    with open(config_path, 'r') as f:
        new_config = yaml.full_load(f)
    config = {}
    if 'include' in new_config:
        include_config = get_config(new_config['include'])
        config.update(include_config)
        del new_config['include']
    config.update(new_config)
    return config


def get_traj_file_name(config):
    if config["env"] == 'metaworld':
        task = config['task']
    else:
        task = config['env']

    path = f'history_{task}_{config["alg"]}_alg-seed{config["alg_seed"]}'

    return path


def ad_collate_fn(batch, grid_size):
    res = {}
    res['query_states'] = torch.tensor(np.array([item['query_states'] for item in batch]), requires_grad=False, dtype=torch.float)
    res['query_goals'] = torch.tensor(np.array([item['query_goals'] for item in batch]), requires_grad=False, dtype=torch.long)
    res['target_actions'] = torch.tensor(np.array([item['target_actions'] for item in batch]), requires_grad=False, dtype=torch.long)
    res['states'] = torch.tensor(np.array([item['states'] for item in batch]), requires_grad=False, dtype=torch.float)
    res['actions'] = torch.tensor(np.array([item['actions'] for item in batch]), requires_grad=False, dtype=torch.long)
    res['rewards'] = torch.tensor(np.array([item['rewards'] for item in batch]), dtype=torch.float, requires_grad=False)
    res['next_states'] = torch.tensor(np.array([item['next_states'] for item in batch]), requires_grad=False, dtype=torch.float)

    if 'target_next_states' in batch[0].keys():
        res['target_next_states'] = map_dark_states(
            torch.tensor(np.array([item['target_next_states'] for item in batch]), dtype=torch.long, requires_grad=False),
            grid_size=grid_size,
        )
        res['target_rewards'] = torch.tensor(np.array([item['target_rewards'] for item in batch]), dtype=torch.long, requires_grad=False)

    return res


def get_data_loader(dataset, batch_size, config, shuffle=True):
    collate_fn = partial(ad_collate_fn, grid_size=config['grid_size'])
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=config['num_workers'],
        persistent_workers=config['num_workers'] > 0,
    )


def log_in_context(values: np.ndarray, max_reward: int, episode_length: int, tag: str, title: str, xlabel: str, ylabel: str, step: int, success=None, writer=None) -> None:
    steps = np.arange(1, len(values[0]) + 1) * episode_length
    mean_value = values.mean(axis=0)

    plt.plot(steps, mean_value)

    if success is not None:
        success_rate = success.astype(np.float32).mean(axis=0)

        for i, (xi, yi) in enumerate(zip(steps, mean_value)):
            if (i + 1) % 10 == 0:
                plt.annotate(f'{success_rate[i]:.2f}', (xi, yi))

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.ylim(-max_reward * 0.05, max_reward * 1.05)
    writer.add_figure(f'{tag}/mean', plt.gcf(), global_step=step)
    plt.close()


def next_dataloader(dataloader: DataLoader):
    """
    Makes the dataloader never end when the dataset is exhausted.
    This is done to remove the notion of an 'epoch' and to count only the amount
    of training steps.
    """
    while True:
        for batch in dataloader:
            yield batch

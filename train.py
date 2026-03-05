from datetime import datetime
import argparse
import os
import os.path as path
from glob import glob

from accelerate import Accelerator

import yaml
import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from dataset import ADDataset
from model import MODEL
from utils import get_config, get_data_loader, next_dataloader
from transformers import get_cosine_schedule_with_warmup

import multiprocessing
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Train AD/FlowAD on Darkroom.')
    parser.add_argument('--env-config', type=str, default='./config/env/darkroom.yaml')
    parser.add_argument('--alg-config', type=str, default='./config/algorithm/ppo_darkroom.yaml')
    parser.add_argument('--model-config', type=str, default='./config/model/ad_dr.yaml')
    parser.add_argument('--traj-dir', type=str, default='./datasets')
    parser.add_argument('--run-root', type=str, default='./runs')
    parser.add_argument('--run-name', type=str, default='')
    parser.add_argument('--run-suffix', type=str, default='')
    parser.add_argument('--train-timesteps', type=int, default=None)
    parser.add_argument('--train-source-timesteps', type=int, default=None)
    parser.add_argument('--train-n-stream', type=int, default=None)
    parser.add_argument('--train-batch-size', type=int, default=None)
    parser.add_argument('--test-batch-size', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--mixed-precision', type=str, default='no', choices=['no', 'fp16', 'bf16'])
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
    return parser.parse_args()


def scalar_items(metrics):
    for key, value in metrics.items():
        if key == 'attentions':
            continue
        if torch.is_tensor(value):
            if value.ndim == 0:
                yield key, value.item()
        elif isinstance(value, (int, float)):
            yield key, float(value)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    args = parse_args()

    config = get_config(args.env_config)
    config.update(get_config(args.alg_config))
    config.update(get_config(args.model_config))

    if args.train_timesteps is not None:
        config['train_timesteps'] = args.train_timesteps
    if args.train_source_timesteps is not None:
        config['train_source_timesteps'] = args.train_source_timesteps
    if args.train_n_stream is not None:
        config['train_n_stream'] = args.train_n_stream
    if args.train_batch_size is not None:
        config['train_batch_size'] = args.train_batch_size
    if args.test_batch_size is not None:
        config['test_batch_size'] = args.test_batch_size
    if args.num_workers is not None:
        config['num_workers'] = args.num_workers

    base_run_name = args.run_name or f"{config['model']}-{config['env']}-seed{config['env_split_seed']}"
    if args.run_suffix:
        base_run_name = f"{base_run_name}-{args.run_suffix}"

    log_dir = path.join(args.run_root, base_run_name)
    os.makedirs(log_dir, exist_ok=True)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    writer = SummaryWriter(log_dir, flush_secs=15) if accelerator.is_main_process else None

    config['log_dir'] = log_dir
    config['traj_dir'] = args.traj_dir
    config['mixed_precision'] = args.mixed_precision
    config['device'] = str(accelerator.device)

    print('Using Device: ', config['device'])

    model_name = config['model']
    model = MODEL[model_name](config)

    load_start_time = datetime.now()
    print(f'Data loading started at {load_start_time}')

    train_dataset = ADDataset(config, config['traj_dir'], 'train', config['train_n_stream'], config['train_source_timesteps'])
    test_dataset = ADDataset(config, config['traj_dir'], 'test', 1, config['train_source_timesteps'])

    train_dataloader = get_data_loader(train_dataset, batch_size=config['train_batch_size'], config=config, shuffle=True)
    test_dataloader = get_data_loader(test_dataset, batch_size=config['test_batch_size'], config=config, shuffle=False)

    load_end_time = datetime.now()
    print()
    print(f'Data loading ended at {load_end_time}')
    print(f'Elapsed time: {load_end_time - load_start_time}')

    optimizer = AdamW(model.parameters(), lr=config['lr'], betas=(config['beta1'], config['beta2']), weight_decay=config['weight_decay'])
    lr_sched = get_cosine_schedule_with_warmup(optimizer, config['num_warmup_steps'], config['train_timesteps'])
    step = 0

    ckpt_paths = sorted(glob(path.join(config['log_dir'], 'ckpt-*.pt')))
    if len(ckpt_paths) > 0:
        ckpt_path = ckpt_paths[-1]
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_sched.load_state_dict(ckpt['lr_sched'])
        step = int(ckpt.get('step', 0))
        print(f'Checkpoint loaded from {ckpt_path}')

    model, optimizer, train_dataloader, test_dataloader, lr_sched = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        test_dataloader,
        lr_sched,
    )
    train_dataloader = next_dataloader(train_dataloader)

    if accelerator.is_main_process:
        config_save_path = path.join(config['log_dir'], 'config.yaml')
        with open(config_save_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config, f, sort_keys=True)

    start_time = datetime.now()
    print(f'Trainig started at {start_time}')

    with tqdm(total=config['train_timesteps'], position=0, leave=True, disable=not accelerator.is_main_process) as pbar:
        pbar.update(step)

        while step < config['train_timesteps']:
            batch = next(train_dataloader)
            step += 1

            with accelerator.accumulate(model):
                output = model(batch)
                loss = output.get('loss', output['loss_action'])

                optimizer.zero_grad(set_to_none=True)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                    lr_sched.step()

            if accelerator.is_main_process:
                pbar.set_postfix(loss=loss.item())

            if step % config['summary_interval'] == 0:
                reduced_loss = accelerator.reduce(loss.detach(), reduction='mean')
                if accelerator.is_main_process and writer is not None:
                    writer.add_scalar('train/loss', reduced_loss.item(), step)
                    for key, value in scalar_items(output):
                        value_tensor = torch.tensor(value, device=accelerator.device, dtype=torch.float32)
                        value_mean = accelerator.reduce(value_tensor, reduction='mean').item()
                        writer.add_scalar(f'train/{key}', value_mean, step)
                    writer.add_scalar('train/lr', lr_sched.get_last_lr()[0], step)

            if step % config['eval_interval'] == 0:
                model.eval()
                eval_start_time = datetime.now()
                if accelerator.is_main_process:
                    print(f'Evaluating started at {eval_start_time}')

                metric_sums = {}
                test_cnt = torch.tensor(0.0, device=accelerator.device)
                with torch.no_grad():
                    for batch in test_dataloader:
                        output = model(batch)
                        cnt = torch.tensor(float(len(batch['states'])), device=accelerator.device)
                        cnt = accelerator.reduce(cnt, reduction='sum')
                        test_cnt += cnt

                        for key, value in scalar_items(output):
                            value_tensor = torch.tensor(value, device=accelerator.device, dtype=torch.float32)
                            value_sum = accelerator.reduce(value_tensor * float(len(batch['states'])), reduction='sum')
                            metric_sums[key] = metric_sums.get(key, torch.tensor(0.0, device=accelerator.device)) + value_sum

                if accelerator.is_main_process and writer is not None and test_cnt.item() > 0:
                    for key, value in metric_sums.items():
                        writer.add_scalar(f'test/{key}', (value / test_cnt).item(), step)

                    eval_end_time = datetime.now()
                    print()
                    print(f'Evaluating ended at {eval_end_time}')
                    print(f'Elapsed time: {eval_end_time - eval_start_time}')
                model.train()

            if step % config['ckpt_interval'] == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    ckpt_paths = sorted(glob(path.join(config['log_dir'], 'ckpt-*.pt')))
                    for ckpt_path in ckpt_paths:
                        os.remove(ckpt_path)

                    new_ckpt_path = path.join(config['log_dir'], f'ckpt-{step}.pt')

                    torch.save({
                        'step': step,
                        'config': config,
                        'model': accelerator.unwrap_model(model).state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_sched': lr_sched.state_dict(),
                    }, new_ckpt_path)
                    print(f'\nCheckpoint saved to {new_ckpt_path}')

            if accelerator.is_main_process:
                pbar.update(1)

    if accelerator.is_main_process and writer is not None:
        writer.flush()
        writer.close()

    end_time = datetime.now()
    if accelerator.is_main_process:
        print()
        print(f'Training ended at {end_time}')
        print(f'Elapsed time: {end_time - start_time}')

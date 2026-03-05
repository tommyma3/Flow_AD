# Algorithm_Distillation
Implementation of algorithm distillation on darkroom environments.

## Original Paper
https://arxiv.org/abs/2210.14215

## Training (Server)
The same `train.py` now supports both original `AD` and `FlowAD`.

### 1) Collect source trajectories
```bash
uv run --python python python collect.py \
  --env-config config/env/darkroom.yaml \
  --alg-config config/algorithm/ppo_darkroom.yaml
```

### 2) Train original AD
```bash
uv run --python python python train.py \
  --model-config config/model/ad_dr.yaml
```

### 3) Train FlowAD
```bash
uv run --python python python train.py \
  --model-config config/model/flowad_dr.yaml
```

### 4) Multi-GPU training with Accelerate
```bash
uv run --python python accelerate launch --num_processes 4 train.py \
  --model-config config/model/flowad_dr.yaml \
  --mixed-precision bf16
```

You can switch to original AD by changing only `--model-config` to `config/model/ad_dr.yaml`.

## Evaluation
```bash
uv run --python python python evaluate.py \
  --ckpt-dir runs/FlowAD-darkroom-seed0
```

For original AD checkpoints:
```bash
uv run --python python python evaluate.py \
  --ckpt-dir runs/AD-darkroom-seed0
```

## Results (historical, AD baseline)
Evaluation goals:  [array([4, 2]), array([5, 6]), array([6, 8]), array([7, 2]), array([3, 6]), array([0, 5]), array([5, 8]), array([5, 4])]  
Mean reward per environment: [17.062 17.102 14.094  0.022 16.1   14.434  6.82   0.49 ]  
Overall mean reward:  10.7655  
Std deviation:  7.961595929837183  

### Figures
Training Loss:  
![training_loss](./figs/training_loss.png)

Testing Loss:
![testing_loss](./figs/testing_loss.png)

Learning Rate Schedule
![lr_schedule](./figs/lr_schedule.png)

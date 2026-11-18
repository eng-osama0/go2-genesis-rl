# Go2 Walking with Reinforcement Learning on Genesis

Training a Unitree Go2 quadruped to follow a forward velocity command using PPO
([rsl-rl](https://github.com/leggedrobotics/rsl_rl)) on the
[Genesis](https://github.com/Genesis-Embodied-AI/Genesis) physics simulator.

## Result (256 envs, 300 iterations)

| Metric | Value |
|---|---|
| Total timesteps | 1,843,200 |
| Training time | ~159 s |
| Throughput | ~11,600 steps/s |
| Mean episode length | 788 / 1000 steps |
| Mean total reward | 8.76 |
| Linear velocity tracking | 0.68 |

Hardware: NVIDIA RTX 4060 Laptop GPU (8 GB). Genesis 1.4.6.
The policy is not fully converged after 300 iterations; longer training is expected to improve stability.

## Setup

```bash
pip install genesis-world
pip install rsl-rl-lib==2.2.4 tensorboard
```

## Usage

```bash
python train.py -e go2-walking -B 256 --max_iterations 300
python eval.py -e go2-walking --ckpt 299
tensorboard --logdir logs
```

## Files

- `go2_env.py`: Genesis environment (observations, rewards, resets)
- `configs.py`: environment and PPO configuration
- `train.py`: training script
- `eval.py`: visualize a trained policy
- `benchmark.py`: simulation throughput measurement

## Notes

Genesis's API changes between versions; this code was adapted to 1.4.6
(e.g. `refresh_rate` instead of `max_FPS`, `rendered_envs_idx`, no `substeps` with `dt`).

"""
قياس سرعة المحاكاة (بدون تعلم): إجراءات عشوائية، ثم طباعة النتيجة بصيغة JSON.
شغّله لكل عدد بيئات بشكل منفصل:
    for n in 1 256 1024 4096; do python benchmark.py -B $n; done
"""
import argparse
import json
import subprocess
import time

import torch
import genesis as gs

from configs import get_cfgs
from go2_env import Go2Env


def gpu_mem_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        )
        return int(out.decode().strip().split("\n")[0])
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-B", "--num_envs", type=int, default=1024)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--warmup", type=int, default=50)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gs.init(backend=gs.gpu if device == "cuda" else gs.cpu, logging_level="error")

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    t_build = time.time()
    env = Go2Env(args.num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg,
                 show_viewer=False, device=device)
    env.reset()
    build_s = time.time() - t_build

    def rand_actions():
        return torch.randn((args.num_envs, env.num_actions), device=env.device) * 0.5

    for _ in range(args.warmup):
        env.step(rand_actions())
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(args.steps):
        env.step(rand_actions())
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    print(json.dumps({
        "simulator": "genesis",
        "num_envs": args.num_envs,
        "env_steps_per_s": round(args.num_envs * args.steps / elapsed),
        "build_time_s": round(build_s, 1),
        "gpu_mem_used_mb": gpu_mem_mb(),
        "genesis_version": gs.__version__,
    }))


if __name__ == "__main__":
    main()

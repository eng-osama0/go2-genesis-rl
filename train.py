import argparse
import os
import pickle
import shutil
import time

import torch
import genesis as gs
from rsl_rl.runners import OnPolicyRunner

from configs import get_cfgs, get_train_cfg
from go2_env import Go2Env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-e", "--exp_name", default="go2-walking")
    p.add_argument("-B", "--num_envs", type=int, default=2048)
    p.add_argument("--max_iterations", type=int, default=101)
    p.add_argument("--vis", action="store_true", help="افتح نافذة عرض أثناء التدريب")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gs.init(backend=gs.gpu if device == "cuda" else gs.cpu, logging_level="warning")

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    log_dir = f"logs/{args.exp_name}"
    shutil.rmtree(log_dir, ignore_errors=True)
    os.makedirs(log_dir, exist_ok=True)
    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    env = Go2Env(args.num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg,
                 show_viewer=args.vis, device=device)
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=device)

    t0 = time.time()
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    print(f"وقت التدريب: {time.time() - t0:.1f} ثانية")


if __name__ == "__main__":
    main()

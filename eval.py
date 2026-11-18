import argparse
import os
import pickle

import torch
import genesis as gs
from rsl_rl.runners import OnPolicyRunner

from go2_env import Go2Env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-e", "--exp_name", default="go2-walking")
    p.add_argument("--ckpt", type=int, default=100)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gs.init(backend=gs.gpu if device == "cuda" else gs.cpu, logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(
        open(f"{log_dir}/cfgs.pkl", "rb")
    )
    reward_cfg["reward_scales"] = {}

    env = Go2Env(1, env_cfg, obs_cfg, reward_cfg, command_cfg,
                 show_viewer=True, device=device)
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=device)
    runner.load(os.path.join(log_dir, f"model_{args.ckpt}.pt"))
    policy = runner.get_inference_policy(device=device)

    obs, _ = env.reset()
    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)


if __name__ == "__main__":
    main()

# run_bot.py
# 部署推理 bot
# 用法:
#   python run_bot.py
#   python run_bot.py --model models/ppo_final --render --verbose

import argparse
import json
import time

import numpy as np
from stable_baselines3 import PPO

from d2r_warlock_env import D2RWarlockEnv, GRID_W
from train import D2RCNN

MODEL_PATH = "models/ppo_final"
KEYMAP_PATH = "models/keymap.json"

MOUSE_NAMES = ["无", "左键按住", "左键单击", "右键"]


def format_action(action, idx_to_key):
    grid_idx, mouse_act, key_act = int(action[0]), int(action[1]), int(action[2])
    gy, gx = divmod(grid_idx, GRID_W)
    mouse_str = MOUSE_NAMES[mouse_act]
    key_name = idx_to_key.get(key_act, "")
    key_str = key_name if key_name else "无"
    return f"格({gx},{gy}) {mouse_str} 键={key_str}"


def run_bot(model_path, n_episodes, render, delay, verbose):
    # 加载按键映射
    with open(KEYMAP_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    idx_to_key = {int(k): v for k, v in config["idx_to_key"].items()}

    print("=" * 50)
    print("D2R Warlock Trav Bot")
    print(f"模型: {model_path}")
    print(f"键盘动作: {config['n_key_actions']} 种")
    print(f"运行 {n_episodes} 局")
    print("=" * 50)

    custom_objects = {
        "policy_kwargs": {
            "features_extractor_class": D2RCNN,
            "features_extractor_kwargs": {"features_dim": 512},
        }
    }
    model = PPO.load(model_path, custom_objects=custom_objects)
    print("模型加载成功")

    env = D2RWarlockEnv(mode='run', render_mode="human" if render else None)

    episode_rewards = []
    episode_lengths = []

    try:
        for ep in range(n_episodes):
            obs, _ = env.reset()
            total_reward = 0.0
            steps = 0
            done = False

            print(f"\n--- Episode {ep + 1}/{n_episodes} ---")

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)

                total_reward += reward
                steps += 1
                done = terminated or truncated

                if verbose and steps % 30 == 0:
                    print(f"  step {steps}: {format_action(action, idx_to_key)}  r={total_reward:.1f}")

                if render:
                    env.render()
                if delay > 0:
                    time.sleep(delay)

            episode_rewards.append(total_reward)
            episode_lengths.append(steps)

            milestones = info.get("milestones", [])
            print(f"  奖励: {total_reward:.1f} | 步数: {steps} | 里程碑: {milestones}")

    except KeyboardInterrupt:
        print("\nBot 中断")
    finally:
        env.close()

    if episode_rewards:
        print("\n" + "=" * 50)
        print(f"完成: {len(episode_rewards)} 局")
        print(f"平均奖励: {np.mean(episode_rewards):.1f} +/- {np.std(episode_rewards):.1f}")
        print(f"平均步数: {np.mean(episode_lengths):.0f}")
        print(f"最高奖励: {np.max(episode_rewards):.1f}")
        print("=" * 50)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run_bot(args.model, args.episodes, args.render, args.delay, args.verbose)


if __name__ == "__main__":
    main()

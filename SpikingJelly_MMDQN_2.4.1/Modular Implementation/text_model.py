"""
text_model.py — 修复版推理脚本
用法示例：
    python text_model.py --scenario highway-v0 --mode TTSA --seed 20 --checkpoint 90000 --num_episodes 5
"""

import os
import argparse
import random

import numpy as np
import torch
import gymnasium as gym
import highway_env

from models import TTSA
from utils.lidar2image import LidarToOccupancyGrid

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def set_global_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_training_env_configs(scenario: str):
    if scenario == "highway-v0":
        config_img = {
            "observation": {
                "type": "GrayscaleObservation",
                "observation_shape": (128, 128),
                "stack_size": 1,
                "weights": [0.299, 0.587, 0.114],
                "scaling": 2,
                "centering_position": [0.5, 0.5],
            },
            "action": {"type": "DiscreteMetaAction"},
            "duration": 50,
        }
        config_lidar = {
            "observation": {
                "type": "LidarObservation",
                "features": ["presence", "distance"],
                "cells": 800,
                "maximum_range": 60,
            },
            "action": {"type": "DiscreteMetaAction"},
            "vehicles_density": 1,
            "duration": 50,
        }

    elif scenario == "roundabout-v0":
        _common = {
            "action": {"type": "DiscreteMetaAction", "target_speeds": [0, 8, 16]},
            "incoming_vehicle_destination": None,
            "collision_reward": -1,
            "high_speed_reward": 0.2,
            "right_lane_reward": 0,
            "lane_change_reward": -0.05,
            "screen_width": 600,
            "screen_height": 600,
            "centering_position": [0.5, 0.6],
            "duration": 11,
            "normalize_reward": True,
        }
        config_img = {
            **_common,
            "observation": {
                "type": "GrayscaleObservation",
                "observation_shape": (128, 128),
                "stack_size": 1,
                "weights": [0.299, 0.587, 0.114],
                "scaling": 2,
                "centering_position": [0.5, 0.5],
            },
        }
        config_lidar = {
            **_common,
            "observation": {
                "type": "LidarObservation",
                "features": ["presence", "distance"],
                "cells": 800,
                "maximum_range": 60,
            },
        }
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return config_img, config_lidar


def make_eval_envs(scenario: str, render_mode: str = "human"):
    config_img, config_lidar = get_training_env_configs(scenario)

    if scenario == "highway-v0":
        config_kin = {
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 1,
                "features": ["heading"],
                "absolute": False,
                "order": "sorted",
            },
            "action": {"type": "DiscreteMetaAction"},
            "duration": 50,
        }
        lidar_converter = LidarToOccupancyGrid(output_range=[50, 200], v_max=30)

    elif scenario == "roundabout-v0":
        _common_kin = {
            "action": {"type": "DiscreteMetaAction", "target_speeds": [0, 8, 16]},
            "incoming_vehicle_destination": None,
            "collision_reward": -1,
            "high_speed_reward": 0.2,
            "right_lane_reward": 0,
            "lane_change_reward": -0.05,
            "screen_width": 600,
            "screen_height": 600,
            "centering_position": [0.5, 0.6],
            "duration": 11,
            "normalize_reward": True,
        }
        config_kin = {
            **_common_kin,
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 1,
                "features": ["heading"],
                "absolute": False,
                "order": "sorted",
            },
        }
        lidar_converter = LidarToOccupancyGrid(output_range=[50, 200], v_max=16)

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    env_img   = gym.make(scenario, render_mode=render_mode, config=config_img)
    env_lidar = gym.make(scenario, render_mode=render_mode, config=config_lidar)
    env_kin   = gym.make(scenario, render_mode=render_mode, config=config_kin)

    return env_img, env_lidar, env_kin, lidar_converter, config_lidar


def load_model(scenario: str, mode: str, seed: int,
               checkpoint_epoch: int, device: torch.device) -> torch.nn.Module:

    config_img, config_lidar = get_training_env_configs(scenario)

    tmp_img   = gym.make(scenario, config=config_img)
    tmp_lidar = gym.make(scenario, config=config_lidar)

    if mode == "TTSA":
        net = TTSA.Network(
            env1=tmp_img,
            env2=tmp_lidar,
            device=device,
            depths1=(8, 16, 16),
            depths2=(8, 16, 16),
            final_layer=512,
            num_steps=5,
            scenario=scenario,
            mode=mode,
            seed=seed,
        )
        # 路径与 save() 保持一致：scenario/mode/seed/MM_DTSQN_H_epoch.pth
        ckpt_path = os.path.join(str(scenario), str(mode), str(seed),
                                 f"MM_DTSQN_H_{checkpoint_epoch}.pth")
    else:
        raise NotImplementedError(f"Mode '{mode}' 尚未实现。")

    tmp_img.close()
    tmp_lidar.close()

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"检查点文件未找到：{ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location=device)
    # 修复核心问题: pos_embedding 现在是正常注册的 nn.Parameter，strict=True 可正常加载
    net.load_state_dict(state_dict, strict=True)

    net.to(device)
    net.eval()
    print(f"[INFO] 模型已加载：{ckpt_path}")

    total_params = sum(p.numel() for p in net.parameters())
    param_bytes  = sum(p.numel() * p.element_size() for p in net.parameters())
    buf_bytes    = sum(b.numel() * b.element_size() for b in net.buffers())
    print(f"[INFO] 总参数量：{total_params / 1e6:.2f} M")
    print(f"[INFO] 模型内存：{(param_bytes + buf_bytes) / 1024 ** 2:.2f} MB")

    return net


def evaluate(scenario: str, mode: str, seed: int,
             checkpoint_epoch: int, render: bool = True,
             num_episodes: int = 5):

    set_global_seed(seed)

    render_mode = "human" if render else "rgb_array"
    env_img, env_lidar, env_kin, lidar_converter, config_lidar = \
        make_eval_envs(scenario, render_mode=render_mode)

    net = load_model(scenario, mode, seed, checkpoint_epoch, device)

    num_rays = config_lidar["observation"]["cells"]
    angles   = np.linspace(0, -2 * np.pi, num_rays)

    episode_rewards = []

    for ep in range(num_episodes):
        h_seed = np.random.randint(0, 2 ** 31)

        obs_img,   info = env_img.reset(seed=h_seed)
        obs_lidar, _    = env_lidar.reset(seed=h_seed)
        obs_kin,   _    = env_kin.reset(seed=h_seed)

        heading  = obs_kin[0][0]
        v_vel    = info["speed"]
        occ_grid = lidar_converter.process(obs_lidar, angles, v_vel, heading)
        obs      = [obs_img, occ_grid]

        done, terminated = False, False
        ep_reward, step_count = 0.0, 0

        while not (done or terminated):
            with torch.no_grad():
                obs_t1 = torch.as_tensor(
                    obs[0], dtype=torch.float32, device=device
                ).unsqueeze(0)
                obs_t2 = torch.as_tensor(
                    obs[1], dtype=torch.float32, device=device
                ).unsqueeze(0)
                q_vals = net(obs_t1, obs_t2)
                action = torch.argmax(q_vals, dim=1).item()

            print(f"  Step {step_count:>4d} | action={action} | speed={info['speed']:.2f}")

            new_img,   reward, done, terminated, info = env_img.step(action)
            new_lidar, *_                             = env_lidar.step(action)
            new_kin,   *_                             = env_kin.step(action)

            new_heading  = new_kin[0][0]
            new_v_vel    = info["speed"]
            new_occ_grid = lidar_converter.process(new_lidar, angles, new_v_vel, new_heading)
            obs = [new_img, new_occ_grid]

            ep_reward  += reward
            step_count += 1

        print(f"Episode {ep + 1}/{num_episodes} | reward={ep_reward:.3f} | steps={step_count}")
        episode_rewards.append(ep_reward)

    env_img.close()
    env_lidar.close()
    env_kin.close()

    mean_rew = np.mean(episode_rewards)
    print(f"\n平均奖励（{num_episodes} 回合）：{mean_rew:.3f}")
    return episode_rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估 spikingjelly 版 TTSA 智能体")
    parser.add_argument("--scenario",     type=str, default="highway-v0",
                        choices=["highway-v0", "roundabout-v0"])
    parser.add_argument("--mode",         type=str, default="TTSA",
                        choices=["TTSA"])
    parser.add_argument("--seed",         type=int, required=True)
    parser.add_argument("--checkpoint",   type=int, required=True)
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--no_render",    action="store_true")
    args = parser.parse_args()

    evaluate(
        scenario=args.scenario,
        mode=args.mode,
        seed=args.seed,
        checkpoint_epoch=args.checkpoint,
        render=not args.no_render,
        num_episodes=args.num_episodes,
    )

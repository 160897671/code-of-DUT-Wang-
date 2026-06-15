import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from models import non_spiking
from models import SSA
from models import TTSA
from utils.lidar2image import LidarToOccupancyGrid
import highway_env

import argparse
import os.path
from collections import deque
import numpy as np
import random
import torch
import gymnasium as gym
import msgpack
from msgpack_numpy import patch as msgpack_numpy_patch
import pickle
import glob

msgpack_numpy_patch()
from torch.utils.tensorboard import SummaryWriter
import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

np.seterr(divide='ignore', invalid='ignore')

# -----------------------------
# ARG PARSER
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=int, default=1,
                    help="Number of seeds to run (each seed = run_index * 10)")
parser.add_argument("--scenario", type=str, default="highway-v0",
                    help="Scenario: highway-v0 or roundabout-v0")
parser.add_argument("--mode", type=str, default="non-spiking",
                    help="Mode: non-spiking, SSA, or TTSA")
parser.add_argument("--warm-start", action="store_true", default=False,
                    help="Enable weight hot-start from previous seed")
parser.add_argument("--share-buffer", action="store_true", default=False,
                    help="Enable experience buffer continuation from previous seed")
parser.add_argument("--buffer-fraction", type=float, default=0.25,
                    help="Fraction of old buffer to retain (default: 0.25)")
parser.add_argument("--ewc-lambda", type=float, default=0.0,
                    help="EWC regularization strength (0 = disabled)")
args = parser.parse_args()


# -----------------------------
# HELPER: SET GLOBAL SEEDS
# -----------------------------
def set_global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# =============================
# TIME FORMATTING HELPER
# =============================
def fmt_time(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


# =============================
# EXPERIENCE BUFFER HELPERS
# =============================
def save_buffer(buffer, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(list(buffer), f)

def load_buffer(filepath):
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


# =============================
# EWC HELPERS
# =============================
def compute_fisher_diag(model, replay_buffer, n_samples=200):
    """
    基于回放池中的样本计算 Fisher 信息矩阵对角线。
    """
    model.eval()
    fisher = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher[name] = torch.zeros_like(param)

    if len(replay_buffer) == 0:
        return fisher

    samples = random.sample(replay_buffer, min(n_samples, len(replay_buffer)))
    for obs, action, rew, done, new_obs in samples:
        model.zero_grad()

        obs_t1 = torch.as_tensor(
            np.asarray(obs[0]), dtype=torch.float32, device=device
        ).unsqueeze(0)
        obs_t2 = torch.as_tensor(
            np.asarray(obs[1]), dtype=torch.float32, device=device
        ).unsqueeze(0)

        q_values = model.forward(obs_t1, obs_t2)
        target_q = q_values.clone().detach()
        loss = torch.nn.functional.mse_loss(q_values, target_q)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += param.grad.data.pow(2)

    for name in fisher:
        fisher[name] /= n_samples

    model.train()
    return fisher


# =============================
# NETWORK FACTORY
# =============================
def make_network(net_type, env, env2, device, scenario, mode, seed):
    if net_type == "non-spiking":
        return non_spiking.Network(env, env2, device=device,
                                   depths1=(8, 16, 16),
                                   depths2=(8, 16, 16),
                                   final_layer=512,
                                   scenario=scenario,
                                   mode=mode,
                                   seed=seed)
    elif net_type == "SSA":
        return SSA.Network(env, env2, device=device,
                           depths1=(8, 16, 16),
                           depths2=(8, 16, 16),
                           final_layer=512,
                           num_steps=5,
                           scenario=scenario,
                           mode=mode,
                           seed=seed)
    elif net_type == "TTSA":
        return TTSA.Network(env, env2, device=device,
                            depths1=(8, 16, 16),
                            depths2=(8, 16, 16),
                            final_layer=512,
                            num_steps=5,
                            scenario=scenario,
                            mode=mode,
                            seed=seed)
    else:
        raise ValueError(f"Unknown mode: {net_type}")


# =============================
# 跨种子全局状态
# =============================
prev_model_path  = None
prev_buffer_path = None
prev_fisher      = None
prev_opt_params  = None

# =====================================================
# ==================== MAIN LOOP ======================
# =====================================================
for run_index in range(args.seeds):

    seed = run_index * 10
    print(f"\n{'='*60}")
    print(f"  RUN {run_index + 1}/{args.seeds} — Seed: {seed}")
    print(f"{'='*60}\n")
    set_global_seed(seed)

    # ---- Hyperparameters ----
    GAMMA            = 0.99
    BATCH_SIZE       = 32
    BUFFER_SIZE      = int(5e4)
    MIN_REPLAY_SIZE  = 10000

    if run_index == 0 or not args.warm_start:
        EPSILON_START = 1.0
        EPSILON_END   = 0.1
        EPSILON_DECAY = int(7e4)
        LR = 1e-4
    else:
        EPSILON_START = 0.3
        EPSILON_END   = 0.05
        EPSILON_DECAY = int(3e4)
        LR = 1e-5

    NUM_ENV          = 1
    TARGET_UPDATE_FREQ = 100 // NUM_ENV
    SAVE_INTERVAL    = 5000
    LOG_INTERVAL     = 1000

    # ========== ENV CONFIGS ==========
    if args.scenario == "highway-v0":
        if args.mode == "non-spiking":
            LOG_DIR = f'logs/non_spiking_Highway/_seed_{seed}'
        elif args.mode == "SSA":
            LOG_DIR = f'logs/SSA_Highway/_seed_{seed}'
        elif args.mode == "TTSA":
            LOG_DIR = f'logs/TTSA_Highway/_seed_{seed}'

        config1 = {
            "observation": {
                "type": "LidarObservation",
                "features": ["presence", "distance"],
                "cells": 800,
                "maximum_range": 60
            },
            "action": {"type": "DiscreteMetaAction"},
            "vehicles_density": 1,
            "duration": 50,
        }
        config = {
            "observation": {
                "type": "GrayscaleObservation",
                "observation_shape": (128, 128),
                "stack_size": 1,
                "weights": [0.299, 0.587, 0.114],
                "scaling": 2,
                "centering_position": [.5, .5]
            },
            "action": {"type": "DiscreteMetaAction"},
            "duration": 50,
        }
        config2 = {
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 1,
                "features": ["heading"],
                "absolute": False,
                "order": "sorted"
            },
            "action": {"type": "DiscreteMetaAction"},
            "duration": 50,
        }
        lidar_converter = LidarToOccupancyGrid(output_range=[50, 200], v_max=30)

        env  = gym.make("highway-v0", render_mode="rgb_array", config=config)
        env2 = gym.make("highway-v0", render_mode="rgb_array", config=config1)
        env3 = gym.make("highway-v0", render_mode="rgb_array", config=config2)
        tenv  = gym.make("highway-v0", render_mode="rgb_array", config=config)
        tenv2 = gym.make("highway-v0", render_mode="rgb_array", config=config1)
        tenv3 = gym.make("highway-v0", render_mode="rgb_array", config=config2)

    elif args.scenario == "roundabout-v0":
        if args.mode == "non-spiking":
            LOG_DIR = f'logs/non_spiking_roundabout/_seed_{seed}'
        elif args.mode == "SSA":
            LOG_DIR = f'logs/SSA_roundabout/_seed_{seed}'
        elif args.mode == "TTSA":
            LOG_DIR = f'logs/TTSA_roundabout/_seed_{seed}'

        config1 = {
            "observation": {
                "type": "LidarObservation",
                "features": ["presence", "distance"],
                "cells": 800,
                "maximum_range": 60
            },
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
        config = {
            "observation": {
                "type": "GrayscaleObservation",
                "observation_shape": (128, 128),
                "stack_size": 1,
                "weights": [0.299, 0.587, 0.114],
                "scaling": 2,
                "centering_position": [.5, .5]
            },
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
        config2 = {
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 1,
                "features": ["heading"],
                "absolute": False,
                "order": "sorted"
            },
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
        lidar_converter = LidarToOccupancyGrid(output_range=[50, 200], v_max=16)

        env  = gym.make("roundabout-v0", render_mode="rgb_array", config=config)
        env2 = gym.make("roundabout-v0", render_mode="rgb_array", config=config1)
        env3 = gym.make("roundabout-v0", render_mode="rgb_array", config=config2)
        tenv  = gym.make("roundabout-v0", render_mode="rgb_array", config=config)
        tenv2 = gym.make("roundabout-v0", render_mode="rgb_array", config=config1)
        tenv3 = gym.make("roundabout-v0", render_mode="rgb_array", config=config2)

    # ======= REPLAY BUFFER =======
    replay_buffer  = deque(maxlen=BUFFER_SIZE)
    epinfos_buffer = deque([], maxlen=100)
    rews_buffer_   = []

    summary_writer = SummaryWriter(LOG_DIR)

    # ======= NETWORK INIT =======
    model_dir = os.path.join(args.scenario, args.mode, str(seed))
    os.makedirs(model_dir, exist_ok=True)

    online_net = make_network(args.mode, env, env2, device,
                              args.scenario, args.mode, seed)
    target_net = make_network(args.mode, env, env2, device,
                              args.scenario, args.mode, seed)

    # ---- 热启动：加载上一个种子的权重 ----
    if args.warm_start and run_index > 0 and prev_model_path is not None:
        print(f"[Warm-Start] Loading weights from: {prev_model_path}")
        state_dict = torch.load(prev_model_path, map_location=device)
        online_net.load_state_dict(state_dict)
        target_net.load_state_dict(state_dict)
        if hasattr(online_net, 'fc_out'):
            online_net.fc_out.reset_parameters()
            target_net.fc_out.reset_parameters()
        print("[Warm-Start] Weights loaded. Output layer reset.")

    online_net = online_net.to(device)
    target_net = target_net.to(device)
    target_net.load_state_dict(online_net.state_dict())

    optimizer = torch.optim.Adam(online_net.parameters(), lr=LR)

    # ---- 经验池延续 ----
    if args.share_buffer and run_index > 0 and prev_buffer_path is not None:
        old_transitions = load_buffer(prev_buffer_path)
        if len(old_transitions) > 0:
            keep_size = int(len(old_transitions) * args.buffer_fraction)
            kept = old_transitions[-keep_size:]
            replay_buffer.extend(kept)
            print(f"[Buffer-Share] Loaded {len(kept)} transitions from previous seed.")

    # ---- EWC 初始化 ----
    ewc_fisher = None
    ewc_optimal_params = None
    if args.ewc_lambda > 0 and run_index > 0 and prev_fisher is not None:
        ewc_fisher = prev_fisher
        ewc_optimal_params = prev_opt_params
        print("[EWC] EWC regularisation enabled.")

    # ======= INITIAL RESET =======
    h_seed = np.random.randint(0, 2 ** 31)
    print(f"Episode map seed: {h_seed}")
    obs1, infos = env.reset(seed=h_seed)
    obs2, _     = env2.reset(seed=h_seed)
    obs3, _     = env3.reset(seed=h_seed)

    num_rays    = config1["observation"]["cells"]
    angle_range = [0, -2 * np.pi]
    angles      = np.linspace(angle_range[0], angle_range[1], num_rays)

    heading        = obs3[0][0]
    v_vel          = infos['speed']
    occupancy_grid = lidar_converter.process(obs2, angles, v_vel, heading)
    obsp = occupancy_grid
    obs  = [obs1, obsp]

    episode_count = 0
    t = 0

    # =====================================================
    # ==================== TRAINING LOOP ==================
    # =====================================================
    train_start_time = time.time()
    step_time_start  = time.time()

    for step in range(int(1e5)):

        epsilon = np.interp(step * NUM_ENV,
                            [0, EPSILON_DECAY],
                            [EPSILON_START, EPSILON_END])

        action = online_net.act(obs, epsilon)

        new_obs1, rew, done, termin, infos = env.step(action)
        new_obs2, _, _, _, _               = env2.step(action)
        new_obs3, _, _, _, _               = env3.step(action)

        heading        = new_obs3[0][0]
        v_vel          = infos['speed']
        occupancy_grid = lidar_converter.process(new_obs2, angles, v_vel, heading)
        obsp    = occupancy_grid
        new_obs = [new_obs1, obsp]

        rews_buffer_.append(rew)
        transition = (obs, action, rew, done, new_obs)
        replay_buffer.append(transition)

        obs = new_obs
        t  += 1

        # -------- EPISODE END --------
        if done or termin:
            eprew = sum(rews_buffer_)
            eplen = len(rews_buffer_)
            epinfos_buffer.append({"r": round(eprew, 6), "l": eplen})

            rews_buffer_  = []
            episode_count += 1

            h_seed         = np.random.randint(0, 2 ** 31)
            obs1, infos    = env.reset(seed=h_seed)
            obs2, _        = env2.reset(seed=h_seed)
            obs3, _        = env3.reset(seed=h_seed)

            v_vel          = infos['speed']
            heading        = obs3[0][0]
            occupancy_grid = lidar_converter.process(obs2, angles, v_vel, heading)
            obsp = occupancy_grid
            obs  = [obs1, obsp]
            t    = 0

        # -------- TRAIN NETWORK --------
        if len(replay_buffer) >= MIN_REPLAY_SIZE:
            transitions = random.sample(replay_buffer, BATCH_SIZE)
            loss = online_net.compute_loss(transitions, target_net)

            # EWC 正则项
            if ewc_fisher is not None and ewc_optimal_params is not None:
                ewc_loss = 0.0
                for name, param in online_net.named_parameters():
                    if name in ewc_fisher and name in ewc_optimal_params:
                        f          = ewc_fisher[name]
                        theta_star = ewc_optimal_params[name]
                        ewc_loss  += (f * (param - theta_star).pow(2)).sum()
                loss = loss + (args.ewc_lambda / 2) * ewc_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # -------- TARGET NET UPDATE --------
        if step % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(online_net.state_dict())

        # -------- LOGGING --------
        if step % LOG_INTERVAL == 0:
            rew_mean = np.mean([e['r'] for e in epinfos_buffer]) or 0
            len_mean = np.mean([e['l'] for e in epinfos_buffer]) or 0

            now             = time.time()
            elapsed         = now - train_start_time
            interval_sec    = now - step_time_start
            step_time_start = now

            steps_per_sec   = LOG_INTERVAL / interval_sec if step > 0 else 0.0
            steps_done      = max(step, 1)
            avg_speed       = steps_done / elapsed
            steps_remaining = int(1e5) - step
            eta_sec         = steps_remaining / avg_speed if avg_speed > 0 else 0.0

            print()
            print(f'Seed: {seed}')
            print(f'Step: {step}')
            print(f'Avg Rew: {rew_mean:.4f}')
            print(f'Avg Ep len: {len_mean:.1f}')
            print(f'Episodes: {episode_count}')
            print(f'Elapsed:  {fmt_time(elapsed)}')
            print(f'Speed:    {steps_per_sec:.1f} steps/s')
            print(f'ETA:      {fmt_time(eta_sec)}')

            summary_writer.add_scalar('AvgRew', rew_mean, step)
            summary_writer.add_scalar('AvgLen', len_mean, step)

        # -------- SAVE & TEST --------
        if step % SAVE_INTERVAL == 0 or step == 0:

            test_reward_sum = 0

            for _ in range(20):
                tseed      = int(np.random.randint(0, 2**31))
                o1, tinfo  = tenv.reset(seed=tseed)
                o2, _      = tenv2.reset(seed=tseed)
                o3, _      = tenv3.reset(seed=tseed)

                headingt       = o3[0][0]
                v_velt         = tinfo['speed']
                occupancy_grid = lidar_converter.process(o2, angles, v_velt, headingt)
                opt    = occupancy_grid
                obst   = [o1, opt]

                d        = False
                ttermin  = False
                ep_ret   = 0
                ep_len   = 0

                while not (d or ttermin):
                    a = online_net.act(obst, 0)
                    tnew_obs1, trew, d, ttermin, tinfos = tenv.step(a)
                    tnew_obs2, *_ = tenv2.step(a)
                    tnew_obs3, *_ = tenv3.step(a)

                    headingn       = tnew_obs3[0][0]
                    v_veln         = tinfos['speed']
                    occupancy_grid = lidar_converter.process(tnew_obs2, angles, v_veln, headingn)
                    tobsp = occupancy_grid
                    obst  = [tnew_obs1, tobsp]

                    ep_ret += trew
                    ep_len += 1

                test_reward_sum += ep_ret

            test_reward_sum /= 20
            summary_writer.add_scalar('TestRew', test_reward_sum, step)
            print(f'TestRew: {test_reward_sum:.4f}')
            print('Saving model...')
            online_net.save(step)

    # -------- END OF SEED --------
    seed_total = time.time() - train_start_time
    print()
    print(f'===== Seed {seed} finished — total time: {fmt_time(seed_total)} =====')

    # ======= 保存最终模型 & 经验池 & Fisher =======
    final_model_path = os.path.join(model_dir, 'final_model.pth')
    torch.save(online_net.state_dict(), final_model_path)
    prev_model_path = final_model_path
    print(f"Final model saved to: {final_model_path}")

    if args.share_buffer:
        buffer_path = os.path.join(model_dir, 'replay_buffer.pkl')
        save_buffer(replay_buffer, buffer_path)
        prev_buffer_path = buffer_path
        print(f"Replay buffer saved to: {buffer_path} (size: {len(replay_buffer)})")
    else:
        prev_buffer_path = None

    if args.ewc_lambda > 0:
        print("Computing Fisher diagonal for EWC...")
        prev_fisher = compute_fisher_diag(online_net, list(replay_buffer), n_samples=200)
        prev_opt_params = {
            name: param.clone().detach()
            for name, param in online_net.named_parameters()
        }
        print("EWC Fisher information stored.")
    else:
        prev_fisher     = None
        prev_opt_params = None

    env.close()
    env2.close()
    env3.close()
    tenv.close()
    tenv2.close()
    tenv3.close()
    summary_writer.close()

print("\n" + "="*60)
print("  ALL SEEDS COMPLETED")
print("="*60)

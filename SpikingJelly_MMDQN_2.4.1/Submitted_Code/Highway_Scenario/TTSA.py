import os.path
import os
from tabnanny import NannyNag
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from pandas.core.nanops import nanmax
from sympy.physics.vector.printing import params

from torch import nn
import torch
import gymnasium as gym
from collections import deque
import itertools
import numpy as np
import random
# import ale_py
# from stable_baselines3.common.vec_env import VecTransposeImage
# from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn
# from stable_baselines3.common.env_util import make_atari_env
# from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
# from stable_baselines3.common.vec_env import VecFrameStack
# from stable_baselines3.common.monitor import Monitor
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch as th
import torch.nn as nn
import snntorch

# from pytorch_wrappers import BatchedPytorchFrameStack, PytorchLazyFrames
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from torch.autograd import Variable
#from myutils.LeakyL import Leaky
import snntorch as snn
from snntorch import spikegen
import highway_env
import msgpack
import einops
from msgpack_numpy import patch as msgpack_numpy_patch
msgpack_numpy_patch()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

GAMMA=0.99
BATCH_SIZE=32
BUFFER_SIZE=int(5e4)
MIN_REPLAY_SIZE=10000

EPSILON_START=1.0
EPSILON_END=0.1
NUM_ENV=1
EPSILON_DECAY=int(7e4)
TARGET_UPDATE_FREQ = 100//NUM_ENV
LR= 1e-4
SAVE_PATH= './atari_model.pack'
SAVE_INTERVAL= 5000
LOG_DIR= 'logs/TTSA'
LOG_INTERVAL=1000


#dtermine the seed
seed=0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

os.makedirs('training_model', exist_ok=True)

config1 = {
    "observation": {
        "type": "LidarObservation",
        "features": ["presence", "distance"],
        "cells": 800,
        "maximum_range": 60
    },
    "action": {
        "type": "DiscreteMetaAction"
    }, "vehicles_density":1,
}

config = {
    "observation": {
        "type": "GrayscaleObservation",
        "observation_shape": (128, 128),
        "stack_size": 1,
        "weights": [0.299, 0.587, 0.114],  # weights for RGB conversion
        "scaling": 2,
        "centering_position": [.5,.5]
    }, "action": {
        "type": "DiscreteMetaAction"
    },
}

config2 = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 1,
        "features": ["heading"],
        "absolute": False,
        "order": "sorted"
    },"action": {
        "type": "DiscreteMetaAction"
    },
            }


def draw_centered_vehicle(grid, length_cells, width_cells, heading_rad, value=1):
    """
    Draw a rotated rectangle in the center of the grid.
    """
    H, W = grid.shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0  # center cell coords

    j = np.arange(W)[None, :]   # column indices
    i = np.arange(H)[:, None]   # row indices
    x = j - cx
    y = i - cy

    # Rotation
    c, s = np.cos(heading_rad), np.sin(heading_rad)
    x_loc =  c * x + s * y
    y_loc = -s * x + c * y

    # Half-size
    hx = length_cells / 2.0
    hy = width_cells  / 2.0

    mask = (np.abs(x_loc) <= hx) & (np.abs(y_loc) <= hy)
    grid[mask] = value
    return grid
class LidarToOccupancyGrid:
    def __init__(self, voxel_size=0.5, x_range=(-60, 60), y_range=(-60, 60), max_distance=60):
        self.voxel_size = voxel_size
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.max_distance = max_distance

        self.grid_width = int((self.x_max - self.x_min) / self.voxel_size)
        self.grid_height = int((self.y_max - self.y_min) / self.voxel_size)
        self.occupancy_grid = np.zeros((self.grid_height + 1, self.grid_width + 1), dtype=int)
        print(self.occupancy_grid.shape)

    def process(self, obs, angles, v_vel, heading):
        points = []

        for i, ray in enumerate(obs):
            if ray[0] > 0 and ray[0] != 1:  # presence
                num = int((1 - abs(ray[0])) * 500)
                distance = np.linspace(ray[0], 1, num) * self.max_distance

                x = distance * np.cos(angles[i])
                y = distance * np.sin(angles[i])
                decay = 0.98 ** np.arange(num)
                z = decay * (ray[1]+v_vel)
                #z = decay * distance
                points.append([x, y, z])

        self.occupancy_grid.fill(0)

        # Add ego vehicle's rectangle
        vl = int(2 // self.voxel_size)
        vh = int(5 // self.voxel_size)
        cx, cy = self.grid_width // 2, self.grid_height // 2

        self.occupancy_grid = draw_centered_vehicle(self.occupancy_grid, vl, vh, np.pi - heading, v_vel)


        # Mark LiDAR points
        for pt in points:
            for j in range(len(pt[0])):
                l = int(pt[0][j] // self.voxel_size)
                k = int(pt[1][j] // self.voxel_size)

                l = self.grid_width // 2 + l
                k = self.grid_height // 2 - k

                if 0 <= k < self.grid_height and 0 <= l < self.grid_width:
                    self.occupancy_grid[l, k] =  pt[2][j]

        #occ = self.occupancy_grid[79:163, 99:141]/30
        occ = self.occupancy_grid[50:200,50:200]/ 30
        cmap = plt.get_cmap("viridis")
        rgbocc = np.uint8(cmap(occ)[:,:,:3]*255)



        rgbocc = np.transpose(rgbocc,(2,0,1))



        #occe = np.expand_dims(occ, 0)

        return rgbocc

lidar_converter = LidarToOccupancyGrid()

class TTSA(nn.Module):
    def __init__(self, dim, num_heads, num_steps):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.numsteps = num_steps
        self.scale = 0.18
        self.q_convi = snntorch.Leaky(.6, 1)
        self.q_conv = nn.Linear(dim, dim)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif1 = snntorch.Leaky(.6, 1)
        self.q_lif2 = snntorch.Leaky(.6, 4)
        self.q1_lif = snntorch.Leaky(.6,1)
        self.k_convi = snntorch.Leaky(.6, 1)
        self.k_conv = nn.Linear(dim, dim)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif1 = snntorch.Leaky(.6, 1)
        self.k_lif2 = snntorch.Leaky(.6, 4)
        self.k1_lif = snntorch.Leaky(.6, 1)
        self.v_convi = snntorch.Leaky(.6, 1)
        self.v_conv = nn.Linear(dim, dim)
        self.v_bn = nn.BatchNorm1d(dim)
        self.blif = snntorch.Leaky(.6, 1)
        self.v1_lif = snntorch.Leaky(.6, 1)
        self.attn_lif = snntorch.Leaky(.6,1)
        self.proj_conv = nn.Linear(dim, dim)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = snntorch.Leaky(.6,1)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, q, k, v):
        memq1 = self.q_lif1.reset_mem()
        memq2 = self.q_lif2.reset_mem()
        memk1 = self.k_lif1.reset_mem()
        memk2 = self.k_lif2.reset_mem()
        mematt = self.attn_lif.reset_mem()
        mproj_lif = self.proj_lif.reset_mem()
        mem_bin = self.blif.reset_mem()
        att_spk_out = []
        v_spk_out = []

        for step in range(self.numsteps):
            B, C, N = q[step].shape
            cur_q_conv_out = self.q_conv(q[step])
            q_conv_out1, memq1 = self.q_lif1(cur_q_conv_out, memq1)
            q_conv_out2, memq2 = self.q_lif2(-cur_q_conv_out, memq2)
            q_conv_out = q_conv_out1 - q_conv_out2
            Q = q_conv_out.reshape(B, C, self.num_heads, N // self.num_heads).permute(0, 2, 1, 3).contiguous()
            B, C, N = k[step].shape
            cur_k_conv_out = self.k_conv(k[step])
            k_conv_out1, memk1 = self.k_lif1(cur_k_conv_out, memk1)
            k_conv_out2, memk2 = self.k_lif2(-cur_k_conv_out, memk2)
            k_conv_out = k_conv_out1 - k_conv_out2
            K = k_conv_out.reshape(B, C, self.num_heads, N // self.num_heads).permute(0, 2, 1, 3).contiguous()
            B, C, N = v[step].shape
            cur_v_conv_out = self.v_conv(v[step])
            V = cur_v_conv_out.reshape(B, C, self.num_heads, N // self.num_heads).permute(0, 2, 1, 3).contiguous()
            x = (Q @ K.transpose(-2, -1)) * self.scale
            spkx, mem_bin = self.blif(x, mem_bin)
            x = (spkx @ V)
            B, H, L, Dh = x.shape
            x = x.permute(0, 2, 1, 3).contiguous()
            cur_x = x.view(B, L, H * Dh)
            x, mematt = self.attn_lif(cur_x, mematt)
            x = self.proj_conv(x)
            x = x.permute(0, 2, 1).contiguous()
            cur = self.proj_bn(x)
            cur = cur.permute(0, 2, 1).contiguous()
            spk_proj, mproj_lif = self.proj_lif(cur, mproj_lif)
            att_spk_out.append(spk_proj)
            v_spk_out.append(v)
        return th.stack(att_spk_out, dim=0), th.stack(v_spk_out, dim=0)

class EncoderBlock(nn.Module):
    def __init__(self, latent_size, num_heads, num_steps, device=device):
        super(EncoderBlock, self).__init__()

        self.latent_size = latent_size
        self.num_heads = num_heads
        self.device = device
        self.num_steps = num_steps

        # Normalization layer
        self.norm1 = nn.LayerNorm(self.latent_size)
        self.norm2 = nn.LayerNorm(self.latent_size)
        self.norm3 = nn.LayerNorm(self.latent_size)

        self.multihead = TTSA(self.latent_size, self.num_heads, self.num_steps)

        self.enc_MLP1 = nn.Linear(self.latent_size, self.latent_size*4)
        self.gelu = snntorch.Leaky(.6, 1)
        self.enc_MLP2 = nn.Linear(self.latent_size*4, self.latent_size)


    def forward(self, embedded_patches1 , embedded_patches2):

        mem = self.gelu.reset_mem()
        spk_out = []
        Q = self.norm1(embedded_patches1)
        V = self.norm2(embedded_patches2)
        attention_out = self.multihead(Q, V, V)[0]
        for step in range(self.num_steps):

            first_added = attention_out[step] + embedded_patches1[step]
            first_added = self.norm3(first_added)
            cur = self.enc_MLP1(first_added)
            spk, mem = self.gelu(cur, mem)
            ff_out = self.enc_MLP2(spk)
            final_out = ff_out + first_added
            spk_out.append(final_out)

        return th.stack(spk_out, dim=0)


class InputEmbedding(nn.Module):
    def __init__(self, n_channels, device, latent_size, dim1, dim2, num_step):
        super(InputEmbedding, self).__init__()
        self.latent_size = latent_size
        self.n_channels = n_channels
        self.device = device
        self.input_size = self.n_channels
        self.num_steps = num_step
        self.dim1 = dim1
        self.dim2 = dim2
        # Linear projection
        self.linearProjection = nn.Linear(self.input_size, self.latent_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, self.dim1, self.dim2)).to(self.device)

    def forward(self, input_data):
        input_data = input_data.to(self.device)
        patches1 = einops.rearrange(input_data, 'T b c (h) (w) -> T b h w (c)')

        patches2 = patches1 + self.pos_embedding.unsqueeze(-1)

        patches = einops.rearrange(patches2, 'T b h w (c) -> T b (h w) (c)')
        linear_projection = self.linearProjection(patches).to(self.device)

        return linear_projection

class Network(nn.Module):
    def __init__(self, env1, env2, device, depths1, depths2, final_layer, num_steps):
        super().__init__()
        self.num_actions = env1.action_space.n

        self.outdim = env1.action_space.n
        self.device = device
        self.depths1 = depths1
        self.depths2 = depths2
        self.num_steps = num_steps
        self.final_layer = final_layer
        n_input_channels1 = env1.observation_space.shape[0]
        n_input_channels2 = 3
        self.encLIF1 = snntorch.Leaky(1, .5)
        self.fc11 = nn.Conv2d(n_input_channels1, depths1[0], kernel_size=5, stride=3)
        self.fc21 = snntorch.Leaky(.6, 1)
        self.fc31 = nn.Conv2d(depths1[0], depths1[1], kernel_size=3, stride=2)
        self.fc41 = snntorch.Leaky(.6, 1)
        self.fc51 = nn.Conv2d(depths1[1], depths1[2], kernel_size=3, stride=1)
        self.fc61 = snntorch.Leaky(.6, 1)
        self.emb1 = InputEmbedding(depths1[2], device=device, latent_size=32, dim1=18, dim2=18, num_step = self.num_steps)

        self.encLIF2 = snntorch.Leaky(1, .5)
        self.fc12 = nn.Conv2d(n_input_channels2, depths2[0], kernel_size=7, stride=3)
        self.fc22 = snntorch.Leaky(.6, 1)
        self.fc32 = nn.Conv2d(depths2[0], depths2[1], kernel_size=5, stride=3)
        self.fc42 = snntorch.Leaky(.6, 1)
        self.fc52 = nn.Conv2d(depths2[1], depths2[2], kernel_size=3, stride=1)
        self.fc62 = snntorch.Leaky(.6, 1)
        self.fc72 = nn.Conv2d(depths2[1], depths2[2], kernel_size=3, stride=1)
        self.fc82 = snntorch.Leaky(.6, 1)
        self.emb2 = InputEmbedding(depths2[1], device=device, latent_size=32, dim1=13, dim2=13, num_step = self.num_steps)

        self.cross = EncoderBlock(32, 8, num_steps=self.num_steps, device=device)

        self.fc7 = nn.Flatten()
        self.fc8 = nn.Linear(5408, final_layer)
        self.fc9 = snntorch.Leaky(.6, 1)
        self.fc10 = nn.Linear(final_layer, self.outdim)

    def forward(self, x1, x2):
        x1 = x1 / 255

        x2 = x2 / 255

        mem21 = self.fc21.reset_mem()
        mem41 = self.fc41.reset_mem()
        mem61 = self.fc61.reset_mem()
        mem22 = self.fc22.reset_mem()
        mem42 = self.fc42.reset_mem()
        mem62 = self.fc62.reset_mem()
        mem9 =  self.fc9.reset_mem()
        spk_V = []
        spk_Q = []

        x_1 = spikegen.rate(x1,self.num_steps).to(device=device)
        x_2 = spikegen.rate(x2, self.num_steps).to(device=device)




        for step in range(self.num_steps):


            cur11 = self.fc11(x_1[step])
            spk21, mem21 = self.fc21(cur11, mem21)
            cur31 = self.fc31(spk21)
            spk41, mem41 = self.fc41(cur31, mem41)
            cur51 = self.fc51(spk41)

            spk61, mem61 = self.fc61(cur51, mem61)

            spk_V.append(spk61)

            cur12 = self.fc12(x_2[step])
            spk22, mem22 = self.fc22(cur12, mem22)
            cur32 = self.fc32(spk22)
            spk42, mem42 = self.fc42(cur32, mem42)
            cur52 = self.fc52(spk42)
            spk62, mem62 = self.fc62(cur52, mem62)
            #Q = self.emb2(spk62)
            spk_Q.append(spk62)
        spkV = th.stack(spk_V, dim=0)
        spkQ = th.stack(spk_Q, dim=0)


        embV = self.emb1(spkV)
        embQ = self.emb2(spkQ)

        cros = self.cross(embQ, embV)

        re_cros = einops.rearrange(cros, 'T b (h w) c -> T b (c) h w', h=13)


        out = []

        for step in range(self.num_steps):
            cur7 = self.fc8(self.fc7(re_cros[step]))

            spk9, mem9 = self.fc9(cur7, mem9)

            outatt = self.fc10(spk9)

            out.append(outatt)

        return th.stack(out, dim=0).sum(dim=0)

    def act(self, obses, epsilon):
        obses_t1 = torch.as_tensor(obses[0], dtype=torch.float32, device=self.device).unsqueeze(0)
        obses_t2 = torch.as_tensor(obses[1], dtype=torch.float32, device=self.device).unsqueeze(0)
        q_values = self(obses_t1, obses_t2)
        max_q_indices = torch.argmax(q_values, dim=1)
        actions = max_q_indices.detach().item()



        rnd_sample = random.random()
        if rnd_sample <= epsilon:
            actions = random.randint(0, self.num_actions-1)

        return actions

    def compute_loss(self, transitions, target_net):

        obses1 = np.asarray([t[0][0] for t in transitions])
        obses2 = np.asarray([t[0][1] for t in transitions])
        actions = np.asarray([t[1] for t in transitions])
        rews = np.asarray([t[2] for t in transitions])
        dones = np.asarray([t[3] for t in transitions])
        new_obses1 = np.asarray([t[4][0] for t in transitions])
        new_obses2 = np.asarray([t[4][1] for t in transitions])
        obses_t1 = torch.as_tensor(obses1, dtype=torch.float32, device=self.device)
        obses_t2 = torch.as_tensor(obses2, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(-1)
        rews_t = torch.as_tensor(rews, dtype=torch.float32, device=self.device).unsqueeze(-1)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(-1)
        new_obses_t1 = torch.as_tensor(new_obses1, dtype=torch.float32, device=self.device)
        new_obses_t2 = torch.as_tensor(new_obses2, dtype=torch.float32, device=self.device)

        # Compute Targets
        # targets = r + gamma * target q vals * (1 - dones)
        target_q_values = target_net(new_obses_t1, new_obses_t2)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
        targets = rews_t + GAMMA * (1 - dones_t) * max_target_q_values

        # Compute Loss
        q_values = self(obses_t1, obses_t2)
        action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)
        loss = nn.functional.smooth_l1_loss(action_q_values, targets)
        return loss

    def save(self, epoch):
        print('model saved')
        th.save(self.state_dict(), 'training_model/TTSA_H_' + str(epoch) + '.pth')

    def load(self, epoch):
        print('load model')
        self.load_state_dict(th.load('training_model/TTSA_H_' + str(epoch) + '.pth'))


env = gym.make("highway-v0", render_mode="rgb_array", config=config)
env2= gym.make("highway-v0", render_mode="rgb_array", config=config1)
env3= gym.make("highway-v0", render_mode="rgb_array", config=config2)
replay_buffer = deque(maxlen=BUFFER_SIZE)
epinfos_buffer = deque([], maxlen=100)

tenv = gym.make("highway-v0", render_mode="rgb_array", config=config)
tenv2= gym.make("highway-v0", render_mode="rgb_array", config=config1)
tenv3= gym.make("highway-v0", render_mode="rgb_array", config=config2)



episode_count = 0
summary_writer = SummaryWriter(LOG_DIR)

#seting networks' architecture
online_net = Network(env,env2, device=device, depths1=(8, 16,16), depths2=(8, 16,16), final_layer=512, num_steps=5)
target_net = Network(env,env2, device=device, depths1=(8, 16,16), depths2=(8, 16,16), final_layer=512, num_steps=5)

online_net= online_net.to(device)
target_net=target_net.to(device)

target_net.load_state_dict(online_net.state_dict())
optimizer = torch.optim.Adam(online_net.parameters(), lr=LR)


rews_buffer_=[]


# Main Training Loop
h_seed = np.random.randint(0, 2**63)
obs1, infos = env.reset(seed=h_seed)
obs2, _ = env2.reset(seed=h_seed)
obs3, _ = env3.reset(seed=h_seed)
num_rays = config1["observation"]["cells"]
angle_range = [0, -2 * np.pi]
angles = np.linspace(angle_range[0], angle_range[1], num_rays)
max_distance = 60
heading = obs3[0][0]
v_vel=infos['speed']

occupancy_grid = lidar_converter.process(obs2, angles, v_vel, heading)
obsp = occupancy_grid

obs = [obs1, obsp]

save_test_reward = []
save_test_reward_steps = []
dter = [0,0,0,0]

t = 0


for step in range(int(1e5)):

    epsilon = np.interp(step * NUM_ENV, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
    rnd_sample = random.random()
    action = online_net.act(obs, epsilon)
    new_obs1, rew, done, termin, infos = env.step(action)
    new_obs2, rew2, done2, _, infos2 = env2.step(action)
    new_obs3, _, _, _, _ = env3.step(action)
    heading = new_obs3[0][0]
    v_vel = infos['speed']
    occupancy_grid = lidar_converter.process(new_obs2, angles, v_vel, heading)
    obsp = occupancy_grid

    new_obs = [new_obs1, obsp]



    i=0
    rews_buffer_.append(rew)
    transition = (obs, action, rew, done, new_obs)
    replay_buffer.append(transition)

    obs = new_obs
    t = t+1





    if done or t>=50:



        eprew = sum(rews_buffer_)
        eplen = len(rews_buffer_)

        epinfo = {"r": round(eprew, 6), "l": eplen}
        epinfos_buffer.append(epinfo)


        rews_buffer_=[]
        episode_count+=1
        h_seed = np.random.randint(0, 2 ** 63)

        obs1, infos = env.reset(seed=h_seed)
        obs2, _ = env2.reset(seed=h_seed)
        obs3, _ = env3.reset(seed=h_seed)
        v_vel = infos['speed']
        heading = obs3[0][0]
        occupancy_grid = lidar_converter.process(obs2, angles, v_vel, heading)
        obsp = occupancy_grid
        obs = [obs1, obsp]
        t = 0



    if len(replay_buffer) >= 1000:


        transitions = random.sample(replay_buffer, BATCH_SIZE)
        loss = online_net.compute_loss(transitions, target_net)

    # Gradient Descent
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Update Target Net
    if step % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(online_net.state_dict())

    # Logging
    if step % LOG_INTERVAL == 0:

        rew_mean = np.mean([e['r'] for e in epinfos_buffer]) or 0
        len_mean = np.mean([e['l'] for e in epinfos_buffer]) or 0





        if step==0:
            rew_mean = 0
            len_mean = 0
        print()
        print('Step:', step)
        print('Avg Rew:', rew_mean)
        print('Avg Ep len:', len_mean)
        print('Epizode:', episode_count)

        summary_writer.add_scalar('AvgRew', rew_mean, global_step=step)
        summary_writer.add_scalar('AvgLen', len_mean, global_step=step)



    if step % SAVE_INTERVAL==0:

        test_reward_sum = 0



        for ij in range(20):
            tseed = np.random.randint(0, 2 ** 63)
            o1, tinfo = tenv.reset(seed=tseed)
            o2, _ = tenv2.reset(seed=tseed)
            o3, _ = tenv3.reset(seed=tseed)

            v_velt = tinfo['speed']
            headingt = o3[0][0]
            occupancy_grid = lidar_converter.process(o2, angles, v_velt, headingt)
            opt = occupancy_grid
            obst = [o1, opt]
            d = False
            ep_len = 0
            ep_ret = 0

            while not (d or (ep_len == 50)):
                a = online_net.act(obst, 0)
                tnew_obs1, trew, d, termin, tinfos = tenv.step(a)
                tnew_obs2, trew2, tdone2, _, tinfos2 = tenv2.step(a)
                tnew_obs3, _, _, _, _ = tenv3.step(a)
                theading = tnew_obs3[0][0]
                tv_vel = tinfos['speed']
                occupancy_grid = lidar_converter.process(tnew_obs2, angles, tv_vel, theading)
                tobsp = occupancy_grid
                tnew_obs = [tnew_obs1, tobsp]
                ep_ret += trew
                ep_len += 1
                obst = tnew_obs
            test_reward_sum += ep_ret
        test_reward_sum = test_reward_sum / 20
        summary_writer.add_scalar('TestRew', test_reward_sum, global_step=step)
        print('TestRew', test_reward_sum)
        print('Saving...')
        online_net.save(step)
import torch
import os
import torch.nn as nn
import einops
import numpy as np
import random
import snntorch
from snntorch import spikegen

GAMMA=0.99
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')





class SSA(nn.Module):
    def __init__(self, dim, num_heads, num_steps, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.numsteps = num_steps
        self.scale = 0.18
        self.q_conv = nn.Linear(dim, dim)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = snntorch.Leaky(.6, 1)

        self.k_conv = nn.Linear(dim, dim)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = snntorch.Leaky(.6, 1)

        self.v_conv = nn.Linear(dim, dim)
        self.v_bn = nn.BatchNorm1d(dim)
        self.blif = snntorch.Leaky(.6, 1)
        self.v_lif = snntorch.Leaky(.6, 1)
        self.attn_lif = snntorch.Leaky(.6, 1)

        self.proj_conv = nn.Linear(dim, dim)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = snntorch.Leaky(.6, 1)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, q, k, v):
        memq = self.q_lif.reset_mem()
        memk = self.k_lif.reset_mem()
        memv = self.v_lif.reset_mem()
        mematt = self.attn_lif.reset_mem()
        mproj_lif = self.proj_lif.reset_mem()
        mem_bin = self.blif.reset_mem()
        att_spk_out = []
        v_spk_out = []

        for step in range(self.numsteps):
            # Q = self.norm1(q[step])
            # B, C, N = Q.shape
            B, C, N = q[step].shape

            q_conv_out = self.q_conv(q[step])
            # q_conv_out = q_conv_out.permute(0,2,1).contiguous()
            # q_conv_out = self.q_bn(q_conv_out)
            # q_conv_out = q_conv_out.permute(0, 2, 1).contiguous()
            q_conv_out, memq = self.q_lif(q_conv_out, memq)
            Q = q_conv_out.reshape(B, C, self.num_heads, N // self.num_heads).permute(0, 2, 1, 3).contiguous()

            # K = self.norm1(k[step])
            # B, C, N = K.shape
            B, C, N = k[step].shape
            k_conv_out = self.k_conv(k[step])
            # k_conv_out = k_conv_out.permute(0, 2, 1).contiguous()
            # k_conv_out = self.k_bn(k_conv_out)
            # k_conv_out = k_conv_out.permute(0, 2, 1).contiguous()
            k_conv_out, memk = self.k_lif(k_conv_out, memk)
            K = k_conv_out.reshape(B, C, self.num_heads, N // self.num_heads).permute(0, 2, 1, 3).contiguous()

            # V = self.norm1(v[step])
            # B, C, N = V.shape
            B, C, N = v[step].shape

            v_conv_out = self.v_conv(v[step])

            # v_conv_out = v_conv_out.permute(0, 2, 1).contiguous()

            # v_conv_out = self.v_bn(v_conv_out)

            v_conv_out, memv = self.v_lif(v_conv_out, memv)
            # v_conv_out = v_conv_out.permute(0, 2, 1).contiguous()

            V = v_conv_out.reshape(B, C, self.num_heads, N // self.num_heads).permute(0, 2, 1, 3).contiguous()

            x = (Q @ K.transpose(-2, -1)) * self.scale

            # shape_x = x.shape
            # x = x.flatten(2,3)

            # spkx, mem_bin = self.blif(x, mem_bin)

            # x = spkx.view(shape_x)

            x = (x @ V)

            B, H, L, Dh = x.shape
            x = x.permute(0, 2, 1, 3).contiguous()
            x = x.view(B, L, H * Dh)

            x, mematt = self.attn_lif(x, mematt)

            # x = x.flatten(0,1)

            x = self.proj_conv(x)

            x = x.permute(0, 2, 1).contiguous()

            cur = self.proj_bn(x)

            cur = cur.permute(0, 2, 1).contiguous()

            spk_proj, mproj_lif = self.proj_lif(cur, mproj_lif)

            att_spk_out.append(spk_proj)
            v_spk_out.append(v)

        return torch.stack(att_spk_out, dim=0), torch.stack(v_spk_out, dim=0)

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

        self.multihead = SSA(self.latent_size, self.num_heads, self.num_steps)

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

        return torch.stack(spk_out, dim=0)


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
    def __init__(self, env1, env2, device, depths1, depths2, final_layer, num_steps, scenario, mode, seed):
        super().__init__()
        self.num_actions = env1.action_space.n

        self.outdim = env1.action_space.n
        self.device = device
        self.depths1 = depths1
        self.depths2 = depths2
        self.num_steps = num_steps
        self.final_layer = final_layer
        self.mode = mode
        self.seed = seed
        self.scenario = scenario
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
        spkV = torch.stack(spk_V, dim=0)
        spkQ = torch.stack(spk_Q, dim=0)


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

        return torch.stack(out, dim=0).sum(dim=0)

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
        torch.save(self.state_dict(), self.scenario + '/' + '-' + self.mode + '/'+ '-' + str(self.seed)+'/MM_DSQN_H_' + str(epoch) + '.pth')

    def load(self, epoch):
        print('load model')
        self.load_state_dict(torch.load(self.scenario + '/' + '-' + self.mode + '/'+ '-' + str(self.seed)+'/MM_DSQN_H_' + str(epoch) + '.pth'))
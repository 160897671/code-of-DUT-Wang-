import torch
import os
import torch.nn as nn
import einops
import numpy as np
import random

# spikingjelly imports
from spikingjelly.activation_based import neuron, functional

GAMMA = 0.99
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------------------
# Helper: build a LIFNode from beta / threshold
#   spikingjelly tau = 1 / (1 - beta)
# ---------------------------------------------------------------------------
def make_lif(beta: float = 0.6, threshold: float = 1.0) -> neuron.LIFNode:
    tau = 1.0 / (1.0 - beta)
    return neuron.LIFNode(tau=tau, v_threshold=float(threshold), step_mode='s')


# ---------------------------------------------------------------------------
# TTSA  (Temporal-domain Spiking Self-Attention)
# ---------------------------------------------------------------------------
class TTSA(nn.Module):
    def __init__(self, dim, num_heads, num_steps):
        super().__init__()
        assert dim % num_heads == 0, \
            f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.numsteps = num_steps
        self.scale = 0.18

        # Q branch
        self.q_conv = nn.Linear(dim, dim)
        self.q_bn   = nn.BatchNorm1d(dim)
        self.q_lif1 = make_lif(0.6, 1)
        self.q_lif2 = make_lif(0.6, 4)

        # K branch
        self.k_conv = nn.Linear(dim, dim)
        self.k_bn   = nn.BatchNorm1d(dim)
        self.k_lif1 = make_lif(0.6, 1)
        self.k_lif2 = make_lif(0.6, 4)

        # V branch
        self.v_conv = nn.Linear(dim, dim)
        self.v_bn   = nn.BatchNorm1d(dim)

        # Attention gate & projection
        self.blif      = make_lif(0.6, 1)
        self.attn_lif  = make_lif(0.6, 1)
        self.proj_conv = nn.Linear(dim, dim)
        self.proj_bn   = nn.BatchNorm1d(dim)
        self.proj_lif  = make_lif(0.6, 1)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, q, k, v):
        """
        q, k, v : lists of length T, each element shape (B, N, C)
        Returns:
            att_spk_out : Tensor [T, B, N, C]
            v_spk_out   : Tensor [T, B, N, C]
        """
        att_spk_out = []
        v_spk_out   = []

        for step in range(self.numsteps):
            B, N, C = q[step].shape

            # ---- Q: bipolar encoding ----
            cur_q = self.q_conv(q[step])          # (B, N, C)
            q_pos = self.q_lif1(cur_q)
            q_neg = self.q_lif2(-cur_q)
            q_out = q_pos - q_neg                  # (B, N, C)
            # reshape: (B, N, num_heads, head_dim) -> (B, num_heads, N, head_dim)
            Q = q_out.reshape(B, N, self.num_heads, C // self.num_heads) \
                     .permute(0, 2, 1, 3).contiguous()

            # ---- K: bipolar encoding ----
            B, N, C = k[step].shape
            cur_k = self.k_conv(k[step])
            k_pos = self.k_lif1(cur_k)
            k_neg = self.k_lif2(-cur_k)
            k_out = k_pos - k_neg
            K = k_out.reshape(B, N, self.num_heads, C // self.num_heads) \
                     .permute(0, 2, 1, 3).contiguous()

            # ---- V: linear projection only ----
            B, N, C = v[step].shape
            cur_v = self.v_conv(v[step])
            V = cur_v.reshape(B, N, self.num_heads, C // self.num_heads) \
                     .permute(0, 2, 1, 3).contiguous()

            # ---- Spiking attention ----
            x    = (Q @ K.transpose(-2, -1)) * self.scale   # (B, H, N, N)
            spkx = self.blif(x)
            x    = spkx @ V                                  # (B, H, N, head_dim)

            B, H, L, Dh = x.shape
            x     = x.permute(0, 2, 1, 3).contiguous()
            cur_x = x.view(B, L, H * Dh)                    # (B, N, C)
            x     = self.attn_lif(cur_x)

            # ---- Projection ----
            x   = self.proj_conv(x)                          # (B, N, C)
            x   = x.permute(0, 2, 1).contiguous()           # (B, C, N) for BN
            cur = self.proj_bn(x)
            cur = cur.permute(0, 2, 1).contiguous()         # (B, N, C)
            spk_proj = self.proj_lif(cur)

            att_spk_out.append(spk_proj)
            v_spk_out.append(v[step])

        return torch.stack(att_spk_out, dim=0), torch.stack(v_spk_out, dim=0)


# ---------------------------------------------------------------------------
# EncoderBlock  (cross-modal attention + feed-forward)
# ---------------------------------------------------------------------------
class EncoderBlock(nn.Module):
    def __init__(self, latent_size, num_heads, num_steps, device=device):
        super().__init__()
        self.latent_size = latent_size
        self.num_heads   = num_heads
        self.device      = device
        self.num_steps   = num_steps

        self.norm1 = nn.LayerNorm(latent_size)
        self.norm2 = nn.LayerNorm(latent_size)
        self.norm3 = nn.LayerNorm(latent_size)

        self.multihead = TTSA(latent_size, num_heads, num_steps)

        self.enc_MLP1 = nn.Linear(latent_size, latent_size * 4)
        self.gelu     = make_lif(0.6, 1)
        self.enc_MLP2 = nn.Linear(latent_size * 4, latent_size)

    def forward(self, embedded_patches1, embedded_patches2):
        """
        embedded_patches1 / 2 : Tensor [T, B, num_patches, latent_size]
        Returns: Tensor [T, B, num_patches, latent_size]
        """
        # norm 对每个时间步的最后一维做，直接作用于 4D tensor 是安全的
        Q = self.norm1(embedded_patches1)   # [T, B, N, C]
        V = self.norm2(embedded_patches2)   # [T, B, N, C]

        # 拆成 list of (B, N, C) 传给 TTSA
        Q_list = [Q[t] for t in range(self.num_steps)]
        V_list = [V[t] for t in range(self.num_steps)]

        attention_out = self.multihead(Q_list, V_list, V_list)[0]  # [T, B, N, C]

        spk_out = []
        for step in range(self.num_steps):
            first_added = attention_out[step] + embedded_patches1[step]
            first_added = self.norm3(first_added)
            cur     = self.enc_MLP1(first_added)
            spk     = self.gelu(cur)
            ff_out  = self.enc_MLP2(spk)
            final_out = ff_out + first_added
            spk_out.append(final_out)

        return torch.stack(spk_out, dim=0)


# ---------------------------------------------------------------------------
# InputEmbedding  (patch projection + positional embedding)
# ---------------------------------------------------------------------------
class InputEmbedding(nn.Module):
    def __init__(self, n_channels, device, latent_size, dim1, dim2, num_step):
        super().__init__()
        self.latent_size = latent_size
        self.n_channels  = n_channels
        self.device      = device
        self.input_size  = n_channels
        self.num_steps   = num_step
        self.dim1 = dim1
        self.dim2 = dim2

        self.linearProjection = nn.Linear(self.input_size, self.latent_size)

        self.pos_embedding = nn.Parameter(
            torch.randn(1, 1, self.dim1, self.dim2)
        )

    def forward(self, input_data):
        input_data = input_data.to(self.device)
        # input_data: (T, B, C, H, W)
        patches1 = einops.rearrange(input_data, 'T b c h w -> T b h w c')
        # pos_embedding: (1, 1, H, W) -> unsqueeze(-1) -> (1, 1, H, W, 1)，broadcast 到 (T, B, H, W, C)
        patches2 = patches1 + self.pos_embedding.unsqueeze(-1)
        patches  = einops.rearrange(patches2, 'T b h w c -> T b (h w) c')
        return self.linearProjection(patches).to(self.device)


# ---------------------------------------------------------------------------
# Network  (full multimodal spiking DQN)
# ---------------------------------------------------------------------------
class Network(nn.Module):
    def __init__(self, env1, env2, device, depths1, depths2,
                 final_layer, num_steps, scenario, mode, seed):
        super().__init__()
        self.num_actions = env1.action_space.n
        self.outdim      = env1.action_space.n
        self.device      = device
        self.depths1     = depths1
        self.depths2     = depths2
        self.num_steps   = num_steps
        self.final_layer = final_layer
        self.mode        = mode
        self.seed        = seed
        self.scenario    = scenario

        n_in1 = env1.observation_space.shape[0]
        n_in2 = 3  # occupancy RGB

        # ---- V stream: RGB camera ----
        self.fc11 = nn.Conv2d(n_in1, depths1[0], kernel_size=5, stride=3)
        self.fc21 = make_lif(0.6, 1)
        self.fc31 = nn.Conv2d(depths1[0], depths1[1], kernel_size=3, stride=2)
        self.fc41 = make_lif(0.6, 1)
        self.fc51 = nn.Conv2d(depths1[1], depths1[2], kernel_size=3, stride=1)
        self.fc61 = make_lif(0.6, 1)
        self.emb1 = InputEmbedding(
            depths1[2], device=device, latent_size=32,
            dim1=18, dim2=18, num_step=num_steps
        )

        # ---- Q stream: LiDAR occupancy ----
        self.fc12 = nn.Conv2d(n_in2, depths2[0], kernel_size=7, stride=3)
        self.fc22 = make_lif(0.6, 1)
        self.fc32 = nn.Conv2d(depths2[0], depths2[1], kernel_size=5, stride=3)
        self.fc42 = make_lif(0.6, 1)
        self.fc52 = nn.Conv2d(depths2[1], depths2[2], kernel_size=3, stride=1)
        self.fc62 = make_lif(0.6, 1)
        # fc72/fc82 保留以兼容旧权重文件，但不参与 forward
        self.fc72 = nn.Conv2d(depths2[1], depths2[2], kernel_size=3, stride=1)
        self.fc82 = make_lif(0.6, 1)
        self.emb2 = InputEmbedding(
            depths2[1], device=device, latent_size=32,
            dim1=13, dim2=13, num_step=num_steps
        )

        # ---- Cross-modal Transformer ----
        self.cross = EncoderBlock(32, 8, num_steps=num_steps, device=device)

        # ---- FC head ----
        self.fc7  = nn.Flatten()
        self.fc8  = nn.Linear(5408, final_layer)
        self.fc9  = make_lif(0.6, 1)
        self.fc10 = nn.Linear(final_layer, self.outdim)

    def forward(self, x1, x2):
        """
        x1 : (B, C, H, W)  RGB camera frame,     pixel values 0-255
        x2 : (B, 3, H, W)  occupancy RGB image,  pixel values 0-255
        Returns: Q-value tensor (B, n_actions)
        """
        x1 = x1 / 255.0
        x2 = x2 / 255.0

        functional.reset_net(self)

        # ---- Poisson rate encoding ----
        x1_exp     = x1.unsqueeze(0).expand(self.num_steps, -1, -1, -1, -1)
        x2_exp     = x2.unsqueeze(0).expand(self.num_steps, -1, -1, -1, -1)
        x1_spk_all = torch.bernoulli(x1_exp)   # (T, B, C, H, W)
        x2_spk_all = torch.bernoulli(x2_exp)

        spk_V = []
        spk_Q = []

        for step in range(self.num_steps):
            # V stream
            cur11 = self.fc11(x1_spk_all[step])
            spk21 = self.fc21(cur11)
            cur31 = self.fc31(spk21)
            spk41 = self.fc41(cur31)
            cur51 = self.fc51(spk41)
            spk61 = self.fc61(cur51)
            spk_V.append(spk61)

            # Q stream
            cur12 = self.fc12(x2_spk_all[step])
            spk22 = self.fc22(cur12)
            cur32 = self.fc32(spk22)
            spk42 = self.fc42(cur32)
            cur52 = self.fc52(spk42)
            spk62 = self.fc62(cur52)
            spk_Q.append(spk62)

        spkV = torch.stack(spk_V, dim=0)   # (T, B, C, H, W)
        spkQ = torch.stack(spk_Q, dim=0)

        embV = self.emb1(spkV)   # (T, B, N, latent)
        embQ = self.emb2(spkQ)

        cros = self.cross(embQ, embV)   # (T, B, N, latent)

        re_cros = einops.rearrange(cros, 'T b (h w) c -> T b c h w', h=13)

        out = []
        for step in range(self.num_steps):
            cur7   = self.fc8(self.fc7(re_cros[step]))
            spk9   = self.fc9(cur7)
            outatt = self.fc10(spk9)
            out.append(outatt)

        return torch.stack(out, dim=0).sum(dim=0)

    def act(self, obses, epsilon):
        obses_t1 = torch.as_tensor(
            obses[0], dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        obses_t2 = torch.as_tensor(
            obses[1], dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        q_values = self(obses_t1, obses_t2)
        actions  = torch.argmax(q_values, dim=1).detach().item()

        if random.random() <= epsilon:
            actions = random.randint(0, self.num_actions - 1)

        return actions

    def compute_loss(self, transitions, target_net):
        obses1     = np.asarray([t[0][0] for t in transitions])
        obses2     = np.asarray([t[0][1] for t in transitions])
        actions    = np.asarray([t[1]    for t in transitions])
        rews       = np.asarray([t[2]    for t in transitions])
        dones      = np.asarray([t[3]    for t in transitions])
        new_obses1 = np.asarray([t[4][0] for t in transitions])
        new_obses2 = np.asarray([t[4][1] for t in transitions])

        obses_t1     = torch.as_tensor(obses1,     dtype=torch.float32, device=self.device)
        obses_t2     = torch.as_tensor(obses2,     dtype=torch.float32, device=self.device)
        actions_t    = torch.as_tensor(actions,    dtype=torch.int64,   device=self.device).unsqueeze(-1)
        rews_t       = torch.as_tensor(rews,       dtype=torch.float32, device=self.device).unsqueeze(-1)
        dones_t      = torch.as_tensor(dones,      dtype=torch.float32, device=self.device).unsqueeze(-1)
        new_obses_t1 = torch.as_tensor(new_obses1, dtype=torch.float32, device=self.device)
        new_obses_t2 = torch.as_tensor(new_obses2, dtype=torch.float32, device=self.device)

        # Target Q-values
        with torch.no_grad():
            target_q_values     = target_net(new_obses_t1, new_obses_t2)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
        targets = rews_t + GAMMA * (1 - dones_t) * max_target_q_values

        # Online Q-values
        q_values        = self(obses_t1, obses_t2)
        action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

        loss = nn.functional.smooth_l1_loss(action_q_values, targets)
        return loss

    def save(self, epoch):
        print('Model saved')
        folder = os.path.join(self.scenario, self.mode, str(self.seed))
        os.makedirs(folder, exist_ok=True)
        filename = f'MM_DTSQN_H_{epoch}.pth'
        path = os.path.join(folder, filename)
        torch.save(self.state_dict(), path)

    def load(self, epoch):
        print('Load model')
        path = os.path.join(self.scenario, self.mode, str(self.seed),
                            f'MM_DTSQN_H_{epoch}.pth')
        self.load_state_dict(torch.load(path, map_location=self.device))

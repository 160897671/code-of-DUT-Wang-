import torch
import os
import torch.nn as nn
import einops
import numpy as np
import random

GAMMA=0.99
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



class EncoderBlock(nn.Module):
    def __init__(self, latent_size, num_heads, device=device):
        super(EncoderBlock, self).__init__()

        self.latent_size = latent_size
        self.num_heads = num_heads
        self.device = device

        # Normalization layer
        self.norm1 = nn.LayerNorm(self.latent_size)
        self.norm2 = nn.LayerNorm(self.latent_size)
        self.norm3 = nn.LayerNorm(self.latent_size)

        self.multihead = nn.MultiheadAttention(
            self.latent_size, self.num_heads, batch_first=True)

        self.enc_MLP = nn.Sequential(
            nn.Linear(self.latent_size, self.latent_size * 4),
            nn.GELU(),
            nn.Linear(self.latent_size * 4, self.latent_size))


    def forward(self, embedded_patches1 , embedded_patches2):


        Q = self.norm1(embedded_patches1)
        V = self.norm2(embedded_patches2)
        attention_out = self.multihead(Q, V, V)[0]
        # first residual connection
        first_added = attention_out + embedded_patches1

        secondnorm_out = self.norm3(first_added)
        ff_out = self.enc_MLP(secondnorm_out)

        return ff_out + first_added


class InputEmbedding(nn.Module):
    def __init__(self, n_channels, device, latent_size, dim1, dim2):
        super(InputEmbedding, self).__init__()
        self.latent_size = latent_size
        self.n_channels = n_channels
        self.device = device
        self.input_size = self.n_channels
        self.dim1 = dim1
        self.dim2 = dim2
        # Linear projection
        self.linearProjection = nn.Linear(self.input_size, self.latent_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.dim1, self.dim2)).to(self.device)

    def forward(self, input_data):
        input_data = input_data.to(self.device)

        patches1 = einops.rearrange(input_data, 'b c (h) (w) -> b h w (c)')


        patches2 = patches1 + self.pos_embedding.unsqueeze(-1)

        patches = einops.rearrange(patches2, 'b h w (c) -> b (h w) (c)')
        linear_projection = self.linearProjection(patches).to(self.device)

        return linear_projection

class Network(nn.Module):
    def __init__(self, env1, env2, device, depths1, depths2, final_layer, scenario, mode, seed):
        super().__init__()
        self.num_actions = env1.action_space.n
        self.outdim = env1.action_space.n
        self.device = device
        self.depths1 = depths1
        self.depths2 = depths2
        self.final_layer = final_layer
        n_input_channels1 = env1.observation_space.shape[0]
        n_input_channels2 = 3
        self.mode = mode
        self.seed = seed
        self.scenario = scenario

        self.fc11 = nn.Conv2d(n_input_channels1, depths1[0], kernel_size=5, stride=3)
        self.fc21 = nn.ReLU()
        self.fc31 = nn.Conv2d(depths1[0], depths1[1], kernel_size=3, stride=2)
        self.fc41 = nn.ReLU()
        self.fc51 = nn.Conv2d(depths1[1], depths1[2], kernel_size=3, stride=1)
        self.fc61 = nn.ReLU()
        self.emb1 = InputEmbedding(depths1[2], device=device, latent_size=32, dim1=18, dim2=18)

        self.fc12 = nn.Conv2d(n_input_channels2, depths2[0], kernel_size=7, stride=3)
        self.fc22 = nn.ReLU()
        self.fc32 = nn.Conv2d(depths2[0], depths2[1], kernel_size=5, stride=3)
        self.fc42 = nn.ReLU()
        self.fc52 = nn.Conv2d(depths2[1], depths2[2], kernel_size=3, stride=1)
        self.fc62 = nn.ReLU()
        self.fc72 = nn.Conv2d(depths2[1], depths2[2], kernel_size=3, stride=1)
        self.fc82 = nn.ReLU()
        self.emb2 = InputEmbedding(depths2[1], device=device, latent_size=32, dim1=13, dim2=13)

        self.cross = EncoderBlock(32, 8, device=device)

        self.fc7 = nn.Flatten()
        self.fc8 = nn.Linear(5408, final_layer)
        self.fc9 = nn.ReLU()
        self.fc10 = nn.Linear(final_layer, self.outdim)

    def forward(self, x1, x2):
        x1 = x1 / 255

        x2 = x2 / 255

        V = self.emb1(self.fc61(self.fc51(self.fc41(self.fc31(self.fc21(self.fc11(x1)))))))

        Q = self.emb2(self.fc62(self.fc52(self.fc42(self.fc32(self.fc22(self.fc12(x2)))))))



        cros = self.cross(Q, V)
        re_cros = einops.rearrange(cros, 'b (h w) c -> b (c) h w', h=13)

        out_Att = self.fc10(self.fc9(self.fc8(self.fc7(re_cros))))


        return out_Att

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
        torch.save(self.state_dict(), self.scenario + '/' + '-' + self.mode + '/'+ '-' + str(self.seed)+'/MM_DQN_H_' + str(epoch) + '.pth')

    def load(self, epoch):
        print('load model')
        self.load_state_dict(self.state_dict(), self.scenario + '/' + '-' + self.mode + '/'+ '-' + str(self.seed)+'/MM_DQN_H_' + str(epoch) + '.pth')
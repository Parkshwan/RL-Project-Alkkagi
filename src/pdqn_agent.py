import random, torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
from collections import deque

class Actor(nn.Module):
    """
    상태 ─► 모든 디스크의 연속 파라미터 예측   [B, num_disc, 2]   ∈ (-1,1)
    """
    def __init__(self, num_total_discs):
        super().__init__()
        self.num_total_discs = num_total_discs
        self.per_disc_dim = 4  # (x, y, team, removed)

        # Encoder for each disc
        self.encoder = nn.Sequential(
            nn.Linear(self.per_disc_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )

        # Aggregation
        self.head = nn.Sequential(
            nn.Linear(32 * num_total_discs, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * num_total_discs),  # fx, fy for each disc
            nn.Tanh()  # output range: (-1, 1)
        )

    def forward(self, obs_flat):
        B = obs_flat.size(0)
        discs = obs_flat.view(B, self.num_total_discs, self.per_disc_dim)
        encoded = self.encoder(discs)  # [B, N, 16]
        encoded = encoded.view(B, -1)  # [B, 16*N]
        return self.head(encoded)     # [B, 2*N]

class Critic(nn.Module):
    """
    (상태, one-hot 디스크 index, 파라미터) ─►  Q값  [B]
    """
    def __init__(self, num_total_discs):
        super().__init__()
        self.num_total_discs = num_total_discs
        self.per_disc_dim = 4

        # Encoder for each disc
        self.encoder = nn.Sequential(
            nn.Linear(self.per_disc_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )

        # Action encoder: one-hot(index) + fx + fy
        self.action_encoder = nn.Sequential(
            nn.Linear(num_total_discs + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        # Q-value network
        self.q_net = nn.Sequential(
            nn.Linear(32 * num_total_discs + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, obs_flat, target_index, fx, fy):
        B = obs_flat.size(0)
        discs = obs_flat.view(B, self.num_total_discs, self.per_disc_dim)
        encoded = self.encoder(discs)  # [B, N, 16]
        encoded = encoded.view(B, -1)  # [B, 16*N]

        # One-hot encode index
        index_onehot = F.one_hot(target_index, num_classes=self.num_total_discs).float()
        a_vec = torch.cat([index_onehot, fx.unsqueeze(1), fy.unsqueeze(1)], dim=1)  # [B, num_total_discs + 2]
        a_feat = self.action_encoder(a_vec)  # [B, 64]

        # Combine state + action
        joint = torch.cat([encoded, a_feat], dim=1)  # [B, 16*N + 64]
        return self.q_net(joint)  # [B, 1]

class PDQNAgent:
    def __init__(
        self,
        num_total_discs,
        device="cpu",
        gamma=0.99,
        tau=0.005,
        actor_lr=1e-4,
        critic_lr=1e-3,
        buffer_size=100000,
        batch_size=256,
        max_grad_norm=1.0
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.num_total_discs = num_total_discs

        # Networks
        self.actor = Actor(num_total_discs).to(device)
        self.actor_target = Actor(num_total_discs).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = Critic(num_total_discs).to(device)
        self.critic_target = Critic(num_total_discs).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Replay buffer: (obs, index, fx, fy, reward, next_obs, done)
        self.buffer = deque(maxlen=buffer_size)

    def select_action(self, obs_flat, legal_indices, epsilon=0.1):
        """
        obs_flat: np.array [4*num_total_discs]
        legal_indices: list of int
        """
        obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0).to(self.device)
        action_output = self.actor(obs_tensor).squeeze(0).detach().cpu().numpy()  # [2*num_total_discs]

        # Random exploration: random index + random continuous action
        if random.random() < epsilon:
            index = random.choice(legal_indices)
            fx = random.uniform(-1.0, 1.0)
            fy = random.uniform(-1.0, 1.0)
            return index, fx, fy

        # Greedy: evaluate Q for all legal indices
        obs_batch = obs_tensor.repeat(len(legal_indices), 1)
        idx_tensor = torch.LongTensor(legal_indices).to(self.device)
        # Vectorized extraction from numpy output
        action_np = action_output.reshape(self.num_total_discs, 2)
        fx_vals = [action_np[i, 0] for i in legal_indices]
        fy_vals = [action_np[i, 1] for i in legal_indices]
        fx_tensor = torch.FloatTensor(fx_vals).to(self.device)
        fy_tensor = torch.FloatTensor(fy_vals).to(self.device)

        with torch.no_grad():
            q_values = self.critic(obs_batch, idx_tensor, fx_tensor, fy_tensor).squeeze(1)
        best_idx = int(torch.argmax(q_values).item())
        index = legal_indices[best_idx]
        fx = fx_vals[best_idx]
        fy = fy_vals[best_idx]
        return index, fx, fy

    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        obs, idx, fx, fy, rew, next_obs, done = self.sample()

        # Compute target Q using actor_target
        with torch.no_grad():
            next_action_output = self.actor_target(next_obs)  # [B, 2*num_total_discs]
            idx_scaled = idx * 2
            next_fx = next_action_output.gather(1, idx_scaled.unsqueeze(1)).squeeze(1)
            next_fy = next_action_output.gather(1, (idx_scaled + 1).unsqueeze(1)).squeeze(1)
            target_q = self.critic_target(next_obs, idx, next_fx, next_fy)
            not_done = 1.0 - done
            target = rew + self.gamma * not_done * target_q

        # Critic update
        current_q = self.critic(obs, idx, fx, fy)
        critic_loss = nn.MSELoss()(current_q, target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        # Actor update: freeze critic parameters to prevent their gradients
        for p in self.critic.parameters():
            p.requires_grad = False
        action_output = self.actor(obs)
        idx_scaled = idx * 2
        act_fx = action_output.gather(1, idx_scaled.unsqueeze(1)).squeeze(1)
        act_fy = action_output.gather(1, (idx_scaled + 1).unsqueeze(1)).squeeze(1)
        actor_loss = -self.critic(obs, idx, act_fx, act_fy).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        # Unfreeze critic parameters
        for p in self.critic.parameters():
            p.requires_grad = True

        # Soft update of target networks
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            
    def store(self, transition):
        obs, index, fx, fy, reward, next_obs, done = transition

        for flip_x in [False, True]:
            for flip_y in [False, True]:
                for flip_team in [False, True]:
                    
                    obs_flip = np.copy(obs)
                    next_obs_flip = np.copy(next_obs)

                    for i in range(self.num_total_discs):
                        # x 좌표 반전
                        if flip_x:
                            obs_flip[4 * i + 0] *= -1
                            next_obs_flip[4 * i + 0] *= -1

                        # y 좌표 반전
                        if flip_y:
                            obs_flip[4 * i + 1] *= -1
                            next_obs_flip[4 * i + 1] *= -1

                        # 팀 정보 반전
                        if flip_team:
                            obs_flip[4 * i + 2] = 1 - obs_flip[4 * i + 2]
                            next_obs_flip[4 * i + 2] = 1 - next_obs_flip[4 * i + 2]

                    # fx, fy 반전
                    fx_flip = -fx if flip_x else fx
                    fy_flip = -fy if flip_y else fy

                    new_transition = (obs_flip, index, fx_flip, fy_flip, reward, next_obs_flip, done)
                    self.buffer.append(new_transition)

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        obs, idx, fx, fy, rew, next_obs, done = zip(*batch)

        obs = torch.FloatTensor(obs).to(self.device)
        idx = torch.LongTensor(idx).to(self.device)
        fx = torch.FloatTensor(fx).to(self.device)
        fy = torch.FloatTensor(fy).to(self.device)
        rew = torch.FloatTensor(rew).unsqueeze(1).to(self.device)
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        return obs, idx, fx, fy, rew, next_obs, done
import random, copy, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple

Transition = namedtuple(
    "Transition",
    ("s", "a_idx", "a_cont", "r", "s2", "done", "valid_mask2")
)

# ───────────────────────── Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=5000):
        self.buf = deque(maxlen=capacity)
    def push(self, *args): self.buf.append(Transition(*args))
    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        return Transition(*zip(*batch))
    def __len__(self): return len(self.buf)

# ───────────────────────── 네트워크
def mlp(in_dim, out_dim, hidden=(256, 256)):
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)

class Actor(nn.Module):
    """
    상태 ─► 모든 디스크의 연속 파라미터 예측   [B, num_disc, 2]   ∈ (-1,1)
    """
    def __init__(self, s_dim, num_disc, param_dim=2):
        super().__init__()
        self.num_disc, self.param_dim = num_disc, param_dim
        self.net = mlp(s_dim, num_disc * param_dim)
    def forward(self, s):
        out = self.net(s)
        out = torch.tanh(out)                  # 값 범위 [-1,1]
        return out.view(-1, self.num_disc, self.param_dim)

class Critic(nn.Module):
    """
    (상태, one-hot 디스크 index, 파라미터) ─►  Q값  [B]
    """
    def __init__(self, s_dim, num_disc, param_dim=2):
        super().__init__()
        in_dim = s_dim + num_disc + param_dim
        self.net = mlp(in_dim, 1)
        self.num_disc = num_disc
    def forward(self, s, a_idx, a_cont):
        # s : [B,s_dim]    a_idx : [B] long   a_cont : [B,param_dim]
        one_hot = F.one_hot(a_idx, self.num_disc).float()
        x = torch.cat([s, one_hot, a_cont], dim=-1)
        return self.net(x).squeeze(-1)         # [B]

# ───────────────────────── PDQN Agent
class PDQNAgent:
    def __init__(self, s_dim, num_disc,
                 gamma=0.99, tau=5e-3,
                 actor_lr=1e-4, critic_lr=2e-4,
                 batch_size=256, device="cpu"):
        self.device = device
        self.num_disc = num_disc
        self.batch = batch_size
        self.gamma, self.tau = gamma, tau

        self.actor  = Actor(s_dim, num_disc).to(device)
        self.critic = Critic(s_dim, num_disc).to(device)
        self.t_actor  = copy.deepcopy(self.actor).eval()
        self.t_critic = copy.deepcopy(self.critic).eval()

        self.opt_a = torch.optim.Adam(self.actor.parameters(),  actor_lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), critic_lr)
        self.replay = ReplayBuffer()

    # ε-greedy 선택 (무효 디스크는 마스킹)
    def act(self, state, valid_mask, epsilon):
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            all_params = self.actor(s)[0].cpu().numpy()   # [num_disc,2]

        if np.random.rand() < epsilon:
            idxs = np.flatnonzero(valid_mask)
            if len(idxs) == 0:           # 경기 끝 직전 보호
                a_idx, a_cont = 0, np.zeros(2)
            else:
                a_idx = int(np.random.choice(idxs))
                a_cont = np.random.uniform(-1, 1, size=2)
        else:
            qvals = []
            for i in range(self.num_disc):
                if not valid_mask[i]:
                    qvals.append(-1e9)
                    continue
                q = self.critic(
                    s,
                    torch.tensor([i], device=self.device),
                    torch.tensor(all_params[i], device=self.device).unsqueeze(0)
                ).item()
                qvals.append(q)
            a_idx = int(np.argmax(qvals))
            a_cont = all_params[a_idx]
        return a_idx, a_cont

    def push(self, *args): 
        # from itertools import product

        # def get_transforms():
        #     rotations = {
        #         0: np.array([[1, 0], [0, 1]]),
        #         90: np.array([[0, -1], [1, 0]]),
        #         180: np.array([[-1, 0], [0, -1]]),
        #         270: np.array([[0, 1], [-1, 0]])
        #     }
        #     flips = {
        #         None: np.eye(2),
        #         'x': np.array([[1, 0], [0, -1]]),
        #         'y': np.array([[-1, 0], [0, 1]])
        #     }

        #     transforms = []
        #     for angle, flip_axis in product(rotations.keys(), flips.keys()):
        #         R = rotations[angle]
        #         F = flips[flip_axis]
        #         M = F @ R
        #         transforms.append(M)
        #     # 중복 제거
        #     unique = []
        #     seen = set()
        #     for M in transforms:
        #         key = tuple(M.flatten())
        #         if key not in seen:
        #             seen.add(key)
        #             unique.append(M)
        #     return unique

        # def transform_obs_vector(obs_vec, M):
        #     """
        #     obs_vec: 1D vector of length 30 → (x, y, idx) * 10
        #     M: 2x2 transform matrix
        #     returns: transformed obs_vec (length 30)
        #     """
        #     obs_arr = np.asarray(obs_vec).reshape(-1, 3)        # (10, 3)
        #     coords = obs_arr[:, :2]                             # (10, 2)
        #     idxs = obs_arr[:, 2:]                               # (10, 1)
        #     transformed_coords = coords @ M.T                  # (10, 2)
        #     transformed_obs = np.hstack([transformed_coords, idxs])
        #     return transformed_obs.flatten()

        # def transform_action_vector(action_vec, M):
        #     """
        #     action_vec: shape (2,)
        #     M: 2x2 matrix
        #     returns: transformed action_vec (2,)
        #     """
        #     action = np.asarray(action_vec).reshape(1, 2)
        #     return (action @ M.T).flatten()

        # # Unpack args
        # obs = np.asarray(args[0])     # 1D vector of length 30
        # action = np.asarray(args[2])  # shape (2,)

        # transforms = get_transforms()

        # for M in transforms:
        #     transformed_obs = transform_obs_vector(obs, M)
        #     transformed_action = transform_action_vector(action, M)

        #     new_args = list(args)
        #     new_args[0] = transformed_obs
        #     new_args[2] = transformed_action

        #     self.replay.push(*new_args)
        # if args[3] == 0: # case reward == 0 -> push to replay buffer 10%
        #     if np.random.rand() < 0.1:
        #         self.replay.push(*args)

        # # elif args[3] == -1:
        # #     if np.random.rand() < 0.1:
        # #         self.replay.push(*args)
        # else:
        #     self.replay.push(*args)
        self.replay.push(*args)

    # 타깃 네트워크 soft-update
    def _soft_update(self, net, tgt):
        for p, tp in zip(net.parameters(), tgt.parameters()):
            tp.data.mul_(1 - self.tau); tp.data.add_(self.tau * p.data)

    # 학습 1 step
    def learn(self):
        if len(self.replay) < self.batch: return
        tr = self.replay.sample(self.batch)

        s   = torch.tensor(tr.s,        device=self.device, dtype=torch.float32)
        a_i = torch.tensor(tr.a_idx,    device=self.device, dtype=torch.long)
        a_c = torch.tensor(tr.a_cont,   device=self.device, dtype=torch.float32)
        r   = torch.tensor(tr.r,        device=self.device, dtype=torch.float32)
        ns  = torch.tensor(tr.s2,       device=self.device, dtype=torch.float32)
        done= torch.tensor(tr.done,     device=self.device, dtype=torch.float32)
        mask2 = torch.tensor(tr.valid_mask2, device=self.device, dtype=torch.bool)

        # ─ Critic update
        with torch.no_grad():
            next_params = self.t_actor(ns)           # [B,num_disc,2]
            q_next = torch.full((self.batch,), -1e9, device=self.device)
            for i in range(self.num_disc):
                valid = mask2[:, i]
                if valid.any():
                    q_i = self.t_critic(
                        ns[valid],
                        torch.full_like(a_i[valid], i),
                        next_params[valid, i]
                    )
                    q_next[valid] = q_i
            y = r + self.gamma * (1 - done) * q_next

        q_pred = self.critic(s, a_i, a_c)
        loss_c = F.mse_loss(q_pred, y)
        self.opt_c.zero_grad(); loss_c.backward(); self.opt_c.step()

        # ─ Actor update  (gradient ascent → -mean)
        params = self.actor(s)                       # [B,num_disc,2]
        q_all = []
        for i in range(self.num_disc):
            q_i = self.critic(s, torch.full_like(a_i, i), params[:, i])
            q_all.append(q_i.unsqueeze(-1))
        q_all = torch.cat(q_all, dim=-1)             # [B,num_disc]
        valid_f = mask2.float()
        loss_a = -(q_all * valid_f).sum() / valid_f.sum()
        self.opt_a.zero_grad(); loss_a.backward(); self.opt_a.step()

        self._soft_update(self.actor,  self.t_actor)
        self._soft_update(self.critic, self.t_critic)

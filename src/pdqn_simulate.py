"""
학습 완료한 PDQN을 로드해 알까기 경기를 실시간 시뮬레이션-렌더링합니다.
"""

import time, argparse, torch, numpy as np
from alkkagi_env import AlkkagiEnv
from pdqn_agent  import Actor, Critic                       # 앞서 작성한 네트워크

# ───────────────────────── CLI
parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=5, help="Number of matches")
parser.add_argument("--delay",    type=float, default=1, help="sec per frame")
parser.add_argument("--gpu",      action="store_true", help="Force CUDA if available")
parser.add_argument("--actor_path",  default="ckpt/pdqn_actor.pth")
parser.add_argument("--critic_path", default="ckpt/pdqn_critic.pth")
args = parser.parse_args()

DEVICE = "cuda" if (args.gpu and torch.cuda.is_available()) else "cpu"

# ───────────────────────── Env & NN 로드
NUM_DISC = 4
env = AlkkagiEnv(num_agent_discs=NUM_DISC, num_opponent_discs=NUM_DISC)
obs_dim = env.reset().size   # flatten() 전 길이

actor  = Actor(obs_dim, NUM_DISC).to(DEVICE).eval()
critic = Critic(obs_dim, NUM_DISC).to(DEVICE).eval()
actor .load_state_dict(torch.load(args.actor_path,  map_location=DEVICE))
critic.load_state_dict(torch.load(args.critic_path, map_location=DEVICE))
env.close()                     # 다시 초기화할 예정

def greedy_action(state, valid_mask):
    """critic을 이용해 Q가 가장 큰 디스크를 골라 fully-greedy 행동 반환"""
    s = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        params = actor(s)[0]                         # [num_disc,2]
        q_vals = torch.full((NUM_DISC,), -1e9, device=DEVICE)

        for i in range(NUM_DISC):
            if not valid_mask[i]:
                continue
            q = critic(
                s,
                torch.tensor([i], device=DEVICE),
                params[i].unsqueeze(0)
            )
            q_vals[i] = q

        idx = int(torch.argmax(q_vals).item())
        return idx, params[idx].cpu().numpy()

def naive_opponent(mask):
    valid = np.flatnonzero(mask)
    if len(valid) == 0: return None
    i = int(np.random.choice(valid))
    fx, fy = np.random.uniform(-0.3, 0.3, 2)
    return np.array([i, fx, fy])

# ───────────────────────── Simulation Loop
for ep in range(1, args.episodes + 1):
    obs = env.reset().flatten()
    done, tot_r = False, 0.0
    print(f"\n===== Match {ep} =====")
    env.render(); time.sleep(1)
    
    while not done:
        mask = env._action_mask()
        a_idx, a_cont = greedy_action(obs, mask)
        obs, r, done, info = env.step(np.array([a_idx, *a_cont]), who=0)
        obs = obs.flatten()
        tot_r += r
        env.render(); time.sleep(1)     # 경기 결과 화면 1초 유지
        
        if done:
            break

        # 간단한 상대 수
        opp = naive_opponent(info["action_mask"])
        if opp is not None:
            obs, r_opp, done, info = env.step(opp, who=1)
            obs = obs.flatten()
            tot_r -= r_opp
        env.render(); time.sleep(1)     # 경기 결과 화면 1초 유지
    
    print(f"→ total return {tot_r:.1f}")

env.close()

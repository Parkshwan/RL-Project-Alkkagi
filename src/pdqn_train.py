import numpy as np, random, torch, time
from alkkagi_env import AlkkagiEnv          # 사용자가 이미 업로드한 env
from pdqn_agent import PDQNAgent

# ───────────────────────── 하이퍼파라미터
SEED            = 2025
NUM_DISC        = 4            # 에이전트와 상대 디스크 수 (동일)
EPISODES        = 1000
MAX_AGENT_TURN  = 60           # agent 턴 step 제한
EPS_START, EPS_END, EPS_GAMMA = 1.0, 0.05, 0.995
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ───────────────────────── Env & Agent 초기화
env = AlkkagiEnv(num_agent_discs=NUM_DISC, num_opponent_discs=NUM_DISC)
obs_flat = env.reset().astype(np.float32).flatten()
agent = PDQNAgent(obs_flat.size, NUM_DISC,
                  batch_size=256, device=DEVICE)

def naive_opponent(mask):
    """간단한 랜덤 상대(학습 초기에 난이도 낮춤)."""
    valid = np.flatnonzero(mask)
    if len(valid) == 0: return None
    idx = int(np.random.choice(valid))
    fx, fy = np.random.uniform(-0.3, 0.3, 2)   # 살살 친다
    return np.array([idx, fx, fy])

# ───────────────────────── 학습 루프
eps = EPS_START
for ep in range(1, EPISODES + 1):
    state = env.reset().flatten()
    total_r, agent_turn = 0.0, 0

    while True:
        # ─ Agent 턴
        valid = env._action_mask()
        a_idx, a_cont = agent.act(state, valid, eps)
        nxt_state, reward, done, info = env.step(
            np.array([a_idx, *a_cont]), who=0
        )
        nxt_state = nxt_state.flatten()

        # ─ Opponent 턴 (optional)
        if not done:
            opp = naive_opponent(info["action_mask"])
            if opp is not None:
                nxt_state, opp_r, done, info = env.step(opp, who=1)
                nxt_state = nxt_state.flatten()
                reward -= opp_r          # 상대가 득점 → 내 보상 감소

        agent.push(state, a_idx, a_cont, reward,
                   nxt_state, done, info["action_mask"])
        agent.learn()

        state, total_r = nxt_state, total_r + reward
        agent_turn += 1
        if done or agent_turn >= MAX_AGENT_TURN:
            break

    eps = max(EPS_END, eps * EPS_GAMMA)
    if ep % 100 == 0:
        print(f"[Ep {ep:4d}] return {total_r:6.2f}  ε {eps:.3f}")

torch.save(agent.actor.state_dict(),  "ckpt/pdqn_actor.pth")
torch.save(agent.critic.state_dict(), "ckpt/pdqn_critic.pth")

env.close()

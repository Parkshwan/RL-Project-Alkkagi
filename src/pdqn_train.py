import numpy as np, random, torch, time
from alkkagi_env import AlkkagiEnv          # 사용자가 이미 업로드한 env
from pdqn_agent import PDQNAgent

# ───────────────────────── 하이퍼파라미터
SEED            = 2025
NUM_DISC        = 5            # 에이전트와 상대 디스크 수 (동일)
EPISODES        = 10000
MAX_TURN  = 60                 # agent 턴 step 제한
EPS_START, EPS_END, EPS_GAMMA = 1.0, 0.05, 0.95
TIME_PAST_REWARD = -0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ───────────────────────── Env & Agent 초기화
env = AlkkagiEnv(num_agent_discs=NUM_DISC, num_opponent_discs=NUM_DISC)
obs_flat = env.reset(random=False).astype(np.float32).flatten()
agent = PDQNAgent(obs_flat.size, NUM_DISC,
                  batch_size=256, device=DEVICE)

def naive_opponent(mask):
    """간단한 랜덤 상대(학습 초기에 난이도 낮춤)."""
    valid = np.flatnonzero(mask)
    if len(valid) == 0: return None
    idx = int(np.random.choice(valid))
    fx, fy = np.random.uniform(-0.3, 0.3, 2)   # 살살 친다
    return idx, np.array([fx, fy])

# ───────────────────────── 학습 루프
eps = EPS_START
a_rewards = []
o_rewards = []
for ep in range(1, EPISODES + 1):
    a_state, o_state = env.reset(random=False).flatten(), None
    o_idx, o_cont, o_valid, o_reward, o_done = None, None, None, None, False
    a_total_r, o_total_r, turn = 0, 0, 0
    render = False if ep%1000 else True
    # render = False

    while True:
        # ─ Agent Turn
        if not o_done:
            a_valid = env.get_action_mask(0)
            a_idx, a_cont = agent.act(a_state, a_valid, eps)
            state, a_reward, a_done, _ = env.step(
                np.array([a_idx, *a_cont]), 0, render
            )
            state = state.flatten()
            a_reward += TIME_PAST_REWARD
            a_total_r += a_reward

            if o_state is not None:
                agent.push(o_state, o_idx, o_cont, o_reward,
                        state, o_done, o_valid)
            o_state = np.concatenate([state[NUM_DISC*3:],state[:NUM_DISC*3]]) # moving discs should be in front
        else:
            if o_state is not None:
                # # done and opponent lost
                # if len(env.opponent_discs) == 0:
                #     o_reward -= 10
                #     o_total_r -= 10
                # # done and opponent won
                # else:
                #     o_reward += 10
                #     o_total_r += 10
                agent.push(o_state, o_idx, o_cont, o_reward,
                        a_state, o_done, o_valid)
            break
        
        # ─ Opponent Turn
        if not a_done:   
            o_valid = env.get_action_mask(1)
            o_idx, o_cont = agent.act(o_state, o_valid, eps)
            # o_idx, o_cont = naive_opponent(o_valid)
            state, o_reward, o_done, _ = env.step(
                np.array([o_idx, *o_cont]), 1, render
            )
            state = state.flatten()
            o_reward += TIME_PAST_REWARD
            o_total_r += o_reward
            

            if a_state is not None:
                agent.push(a_state, a_idx, a_cont, a_reward,
                        state, a_done, a_valid)
            a_state = state
        else:
            if a_state is not None:
                # # done and agent lost
                # if len(env.agent_discs) == 0:
                #     a_reward -= 10
                #     a_total_r -= 10
                # # done and agent won
                # else:
                #     a_reward += 10
                #     a_total_r += 10
                agent.push(a_state, a_idx, a_cont, a_reward,
                        o_state, a_done, a_valid)
            break
        
        agent.learn()

        turn += 1
        if turn >= MAX_TURN:
            break
    
    a_rewards.append(a_total_r)
    o_rewards.append(o_total_r)

    if ep % 100 == 0:
        print(f"[Ep {ep:4d}] agent return {sum(a_rewards)/len(a_rewards):6.2f} | opponent return {sum(o_rewards)/len(o_rewards):6.2f} | ε {eps:.3f}")
        a_rewards = []
        o_rewrads = []
        eps = max(EPS_END, eps * EPS_GAMMA)

    if ep % 1000 == 0:
        torch.save(agent.actor.state_dict(),  "ckpt/pdqn_actor.pth")
        torch.save(agent.critic.state_dict(), "ckpt/pdqn_critic.pth")
        print(f"[Ep {ep:4d}] checkpoint saved")

# torch.save(agent.actor.state_dict(),  "ckpt/pdqn_actor.pth")
# torch.save(agent.critic.state_dict(), "ckpt/pdqn_critic.pth")

env.close()

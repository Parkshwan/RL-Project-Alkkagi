# pdqn_train.py
# ──────────────────────────────────────────────────────────────────────
"""
PDQN 학습 스크립트 (1vs1 → 5vs5 커리큘럼)

조건
1) AlkkagiEnv(num_discs_per_player=5, num_curriculum=1)  ← 항상 10 슬롯
2) 에이전트(team==0)가 선공
3) 상대는 휴리스틱(opponent)로 플레이
4) 버퍼에 [s, idx, fx, fy, r, s', done] 저장  (s' = 상대 턴 이후 상태)
"""
import os, random, time
import numpy as np
import torch
from alkkagi_env import AlkkagiEnv
from pdqn_agent import PDQNAgent

# ────────────────────────── 하이퍼파라미터
SEED                  = 2025
EPISODES              = 500_000
MAX_AGENT_TURNS       = 20          # 한 에피소드에서 agent가 가지는 최대 턴 수
CURRICULUM_INTERVAL   = 100_000        # N 에피소드마다 num_curriculum +1
EPS_START, EPS_END    = 1.0, 0.2
EPS_DECAY             = 0.9983
DEVICE                = "cuda" if torch.cuda.is_available() else "cpu"
LOG_EVERY             = 500
SAVE_EVERY            = 10_000

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
os.makedirs("ckpt", exist_ok=True)

# ────────────────────────── 유틸
def flatten_obs(obs_tuple):
    """
    obs[i] = ((x, y), team, removed)  →  [x, y, team, removed] 1차원 배열
    """
    flat = []
    for (x, y), team, removed in obs_tuple:
        flat.extend([x, y, float(team), float(removed)])
    return np.asarray(flat, dtype=np.float32)

def heuristic_opponent(env):
    """
    team==1 돌 하나를 랜덤 선택 → 가장 가까운 agent 돌 조준.
    가우시안 노이즈를 포함해 (-1,1) 범위의 힘 벡터 반환.
    """
    # 아직 남아있는 own / target 돌
    own  = [d for d in env.discs if d.team == 1 and not d.removed]
    targ = [d for d in env.discs if d.team == 0 and not d.removed]
    if len(own) == 0 or len(targ) == 0:
        return None                        # nothing to do

    shooter = random.choice(own)
    # 가장 가까운 상대 찾기
    nearest = min(targ,
                  key=lambda d: np.linalg.norm(np.array(d.position) -
                                              np.array(shooter.position)))

    # 정규화된 방향 + 노이즈
    vec = np.array(nearest.position) - np.array(shooter.position)
    norm = np.linalg.norm(vec)
    direction = vec / norm if norm > 1e-6 else np.random.uniform(-1, 1, 2)
    direction += np.random.normal(0, 0.05, 2)          # Gaussian noise
    direction = np.clip(direction, -1, 1)

    # 힘 세기(0.5~0.8) 스칼라 곱
    power = np.random.uniform(0.6, 0.8)
    direction = direction * power / np.maximum(np.linalg.norm(direction), 1e-6)
    direction = np.clip(direction, -1, 1)              # 재클립

    return int(shooter.index), direction.astype(np.float32)

# ────────────────────────── Env & Agent 초기화
env  = AlkkagiEnv(num_discs_per_player=1, num_curriculum=1, visualize=False)
agent = PDQNAgent(num_total_discs=env.num_total_discs,
                  device=DEVICE,
                  batch_size=256)

eps = EPS_START
returns, curriculum = [], env.num_curriculum

# ────────────────────────── 학습 루프
for ep in range(1, EPISODES + 1):

    obs_tuple   = env.reset()
    state       = flatten_obs(obs_tuple)
    done        = False
    ep_return   = 0.0
    turn_count  = 0

    
    
    while not done and turn_count < MAX_AGENT_TURNS:
        # ───── Agent 턴 ────────────────────────────────────────────────
        legal = env.get_alive_stone_index(0)
        if len(legal) == 0:
            break                                        # 에이전트 돌이 없음

        idx, fx, fy = agent.select_action(state, legal, eps)
        next_obs, reward, done_flag, _ = env.step((idx, (fx, fy)), who=0)
        # 아직 "상대 턴" 전의 상태 → opponent 이후로 업데이트
        opp_obs = next_obs
        done    = done_flag
        
        # ───── Opponent Transition 저장 & 학습 ─────────────────────────────────
        # if turn_count > 0: 
        #     transition = info_opp + (flatten_obs(next_obs), float(done))
        #     agent.store(transition)
        #     agent.update()

        # ───── Opponent 턴 ─────────────────────────────────────────────
        if not done:
            opp_action = heuristic_opponent(env)
            if opp_action is not None:
                opp_idx, opp_dir = opp_action
                opp_obs, opp_reward, done, _ = env.step((opp_idx, opp_dir), who=1)
                info_opp = (
                    flatten_obs(next_obs), opp_idx, opp_dir[0], opp_dir[1], opp_reward
                )

        # ───── Transition 저장 & 학습 ─────────────────────────────────
        transition = (
            state, idx, fx, fy, reward,
            flatten_obs(opp_obs), float(done)
        )
        agent.store(transition)
        agent.update()

        # 다음 스텝 준비
        state      = flatten_obs(opp_obs)
        ep_return += reward
        turn_count += 1
        

    # ───── 에피소드 종료 처리 ─────────────────────────────────────────
    returns.append(ep_return)

    if ep % LOG_EVERY == 0:
        eps = max(EPS_END, eps * EPS_DECAY)
        avg = np.mean(returns[-LOG_EVERY:])
        print(f"[EP {ep:5d}] return:{avg:7.3f}  ε:{eps:6.3f}  "
              f"curr:{env.num_curriculum}")

# ────────────────────────── 모델 저장
    if ep % SAVE_EVERY == 0:
        torch.save(agent.actor.state_dict(),  "ckpt/pdqn_actor.pth")
        torch.save(agent.critic.state_dict(), "ckpt/pdqn_critic.pth")
        
    # 커리큘럼 단계 향상
    if ep % CURRICULUM_INTERVAL == 0 and curriculum < env.num_discs_per_player:
        curriculum += 1
        env.num_curriculum = curriculum
        eps = EPS_START
        print(f"[+] Curriculum up → {curriculum}v{curriculum}")
        
env.close()

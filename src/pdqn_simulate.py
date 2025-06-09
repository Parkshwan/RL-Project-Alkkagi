import time
import torch
import numpy as np
from alkkagi_env import AlkkagiEnv
from pdqn_agent import PDQNAgent

AGENT = 0
OPPONENT = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def flatten_obs(obs_tuple):
    flat = []
    for (x, y), team, removed in obs_tuple:
        flat.extend([x, y, float(team), float(removed)])
    return np.asarray(flat, dtype=np.float32)

def heuristic_opponent(env):
    own = [d for d in env.discs if d.team == 1 and not d.removed]
    targ = [d for d in env.discs if d.team == 0 and not d.removed]
    if len(own) == 0 or len(targ) == 0:
        return None

    shooter = np.random.choice(own)
    nearest = min(targ, key=lambda d: np.linalg.norm(np.array(d.position) - np.array(shooter.position)))
    vec = np.array(nearest.position) - np.array(shooter.position)
    norm = np.linalg.norm(vec)
    direction = vec / norm if norm > 1e-6 else np.random.uniform(-1, 1, 2)
    direction += np.random.normal(0, 0.05, 2)
    direction *= np.random.uniform(0.5, 0.8) / max(np.linalg.norm(direction), 1e-6)
    direction = np.clip(direction, -1, 1)
    return int(shooter.index), direction.astype(np.float32)

def simulate_episode(num_discs_per_player = 5,  num_curriculum = 3, visualize=True):
    env = AlkkagiEnv(
        num_discs_per_player = num_discs_per_player, 
        num_curriculum = num_curriculum, 
        visualize=visualize, 
        fixed=False
    )
    agent = PDQNAgent(num_total_discs=2, device=DEVICE)
    agent.actor.load_state_dict(torch.load("ckpt/pdqn_actor.pth", map_location=DEVICE))
    agent.actor.eval()
    agent.critic.load_state_dict(torch.load("ckpt/pdqn_critic.pth", map_location=DEVICE))
    agent.critic.eval()

    obs = env.reset()
    state = flatten_obs(obs)
    done = False
    step = 1
    
    env.render()
    time.sleep(5)

    while not done:
        # Agent turn
        legal = env.get_alive_stone_index(AGENT)
        if legal:
            idx, fx, fy = agent.select_action(state, legal, epsilon=0.0)
            obs, reward, done, _ = env.step((idx, (fx, fy)), AGENT)
            print(f"[Agent] Step {step} | idx={idx}, fx={fx:.2f}, fy={fy:.2f}, reward={reward:.2f}, done={done}")
            step += 1
            if visualize:
                env.render()
                time.sleep(0.7)

        if done: break

        # Opponent turn
        legal = env.get_alive_stone_index(OPPONENT)
        if legal:
            opp_action = heuristic_opponent(env)
            if opp_action:
                opp_idx, opp_dir = opp_action
                obs, reward, done, _ = env.step((opp_idx, opp_dir), OPPONENT)
                print(f"[Opponent] Step {step} | idx={opp_idx}, reward={reward:.2f}, done={done}")
                step += 1
                if visualize:
                    env.render()
                    time.sleep(0.7)

        if done: break

        state = flatten_obs(obs)

    env.close()

if __name__ == "__main__":
    simulate_episode(num_discs_per_player = 1, num_curriculum = 1)
import time
import numpy as np
from alkkagi_env import AlkkagiEnv

AGENT = 0
OPPONENT = 1

env = AlkkagiEnv(num_discs_per_player=5, visualize=True, fixed=False)
obs = env.reset()

step = 1
env.render()
time.sleep(10)

def passive_random_action(alive_indices):
    if not alive_indices:
        return None
    idx = np.random.choice(alive_indices)
    direction = np.random.uniform(-1.0, 1.0, 2)
    direction = direction / np.linalg.norm(direction + 1e-8)  # 방향 정규화
    power = np.random.uniform(0.05, 0.15)  # 매우 약한 힘 (소극적)
    action_vec = direction * power
    return idx, action_vec

while True:
    # agent turn
    alive_agent = env.get_alive_stone_index(AGENT)
    agent_action = passive_random_action(alive_agent)
    if agent_action is None:
        break
    obs, reward, done, _ = env.step(agent_action, AGENT)
    print(f"step: {step}, AGENT action: {agent_action}, reward: {reward:.3f}, done: {done}")
    step += 1
    env.render()
    time.sleep(1)
    if done:
        break

    # opponent turn
    alive_opponent = env.get_alive_stone_index(OPPONENT)
    opponent_action = passive_random_action(alive_opponent)
    if opponent_action is None:
        break
    obs, reward, done, _ = env.step(opponent_action, OPPONENT)
    print(f"step: {step}, OPPONENT action: {opponent_action}, reward: {reward:.3f}, done: {done}")
    step += 1
    env.render()
    time.sleep(1)
    if done:
        break

env.close()

import time
from alkkagi_env import AlkkagiEnv
import numpy as np

env = AlkkagiEnv(num_agent_discs=4, num_opponent_discs= 4)
obs = env.reset()

action = np.array([0, 0.0, 0.0])  # 위쪽으로 힘 가함
obs, reward, done, _ = env.step(action)

step = 1

while not done:
    env.render()
    obs, reward, done, _ = env.step(np.array([0, 0.0, 0.0]))
    print(f"step: {step}, IsDone: {done}")
    time.sleep(1)
    step += 1

print("최종 보상:", reward)
env.close()

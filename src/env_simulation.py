import time
from alkkagi_env import AlkkagiEnv
import numpy as np

env = AlkkagiEnv()
obs = env.reset()

action = np.array([0.0, 0.0])  # 위쪽으로 힘 가함
obs, reward, done, _ = env.step(action)

step = 1

while not done:
    env.render()
    print(f"step: {step}")
    obs, reward, done, _ = env.step(np.array([0.0, 0.0]))
    time.sleep(0.01)
    step += 1

print("최종 보상:", reward)
env.close()

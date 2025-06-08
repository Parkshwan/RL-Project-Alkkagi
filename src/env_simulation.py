import time
from alkkagi_env import AlkkagiEnv
import numpy as np

env = AlkkagiEnv(num_discs_per_player = 5)
obs = env.reset()

action = np.array([0, 0.0, 0.0])  # 위쪽으로 힘 가함
obs, reward, done, _ = env.step(action, 0)

step = 1
env.render()
time.sleep(1)
while not done:
    # agent turn
    obs, reward, done, _ = env.step(np.array([0, 0, -0.3]), 0)
    print(f"step: {step}, IsDone: {done}, reward: {reward}")
    step += 1
    env.render()
    time.sleep(1)
    
    if done:
         break

    # opponent turn
    obs, reward, done, _ = env.step(np.array([0, 0, 0.1]), 1)
    print(f"step: {step}, IsDone: {done}, reward: {reward}")
    step += 1
    env.render()
    time.sleep(1)

print("최종 보상:", reward)
env.close()

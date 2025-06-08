import time
from alkkagi_env import AlkkagiEnv
import numpy as np

AGENT = 0
OPPONENT = 1

env = AlkkagiEnv(num_discs_per_player = 1)
obs = env.reset()

step = 1
env.render()
time.sleep(1)

while True:
    # agent turn
    alive_agent = env.get_alive_stone_index(AGENT)
    
    obs, reward, done, _ = env.step((alive_agent[0], (0, -0.5)), AGENT)
    print(f"step: {step}, IsDone: {done}, reward: {reward}")
    step += 1
    env.render()
    time.sleep(1)
    
    if done:
         break
    
    # opponent turn
    
    alive_opponent = env.get_alive_stone_index(OPPONENT)
    obs, reward, done, _ = env.step((alive_opponent[0], (0, 0.5)), OPPONENT)
    print(f"step: {step}, IsDone: {done}, reward: {reward}")
    step += 1
    env.render()
    time.sleep(1)
    
    if done:
         break

print("최종 보상:", reward)
env.close()

import numpy as np, random, torch, time
from alkkagi_env import AlkkagiEnv
from pdqn_agent import PDQNAgent
from pdqn_simulate import simulate

# hyperparameters
SEED            = 2025
NUM_DISC        = 5
EPISODES        = 1000000
MAX_TURN        = 20
EPS_START, EPS_END, EPS_GAMMA = 1.0, 0.2, 0.9985
TIME_PAST_REWARD = -1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# initialize environment and model
env = AlkkagiEnv(num_agent_discs=NUM_DISC, num_opponent_discs=NUM_DISC)
obs_flat = env.reset(random=False).astype(np.float32).flatten()
agent = PDQNAgent(obs_flat.size, NUM_DISC,
                  batch_size=256, device=DEVICE, load=False)

def naive_opponent(mask):
    """Simple random opponent."""
    valid = np.flatnonzero(mask)
    if len(valid) == 0: return None
    idx = int(np.random.choice(valid))
    fx, fy = np.random.uniform(-0.7, 0.7, 2) # relatively weak power
    return idx, np.array([fx, fy])

# training loop
eps = EPS_START
a_rewards = []
o_rewards = []
for ep in range(1, EPISODES + 1):
    a_state, o_state = env.reset(random=True).flatten(), None
    o_idx, o_cont, o_valid, o_reward, o_done = None, None, None, None, False
    a_total_r, o_total_r, turn = 0, 0, 0

    while True:
        # agent turn
        if not o_done:
            a_valid = env.get_action_mask(0)
            a_idx, a_cont = agent.act(a_state, a_valid, eps)
            state, a_reward, a_done, _ = env.step(
                np.array([a_idx, *a_cont]), 0
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
                agent.push(o_state, o_idx, o_cont, o_reward,
                        a_state, o_done, o_valid)
            break
        
        # opponent turn
        if not a_done:   
            o_valid = env.get_action_mask(1)
            o_idx, o_cont = agent.act(o_state, o_valid, eps) # train with model-based opponent
            # o_idx, o_cont = naive_opponent(o_valid) # train with simple random opponent
            state, o_reward, o_done, _ = env.step(
                np.array([o_idx, *o_cont]), 1
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
                agent.push(a_state, a_idx, a_cont, a_reward,
                        o_state, a_done, a_valid)
            break
        
        agent.learn()

        turn += 1
        if turn >= MAX_TURN:
            break
    
    a_rewards.append(a_total_r)
    o_rewards.append(o_total_r)

    # printing result and decreasing epsilon(exploration rate)
    if ep % 1000 == 0:
        a_reward_avg = sum(a_rewards)/len(a_rewards)
        o_reward_avg = sum(o_rewards)/len(o_rewards)
        print(f"[Ep {ep:4d}] agent return {a_reward_avg:6.2f} | opponent return {o_reward_avg:6.2f} | ε {eps:.3f}")
        a_rewards = []
        o_rewrads = []
        eps = max(EPS_END, eps * EPS_GAMMA)

    # save checkpoint
    if ep % 10000 == 0:
        torch.save(agent.actor.state_dict(),  f"ckpt/5/pdqn_actor_{ep}.pth")
        torch.save(agent.critic.state_dict(), f"ckpt/5/pdqn_critic_{ep}.pth")
        print(f"[Ep {ep:4d}] periodic checkpoint saved")
        # print(f"[Ep {ep:4d}] simulation result: {simulate():.1f}")

env.close()
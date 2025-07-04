import time, argparse, torch, numpy as np
from alkkagi_env import AlkkagiEnv
from pdqn_agent  import Actor, Critic


def greedy_action(state, valid_mask, actor, critic, DEVICE, NUM_DISC):
    """Use critc to select the disc with the largest Q and return the fully-greedy behavior"""
    s = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        params = actor(s)[0]                         # [num_disc,2]
        q_vals = torch.full((NUM_DISC,), -1e9, device=DEVICE)

        for i in range(NUM_DISC):
            if not valid_mask[i]:
                continue
            q = critic(
                s,
                torch.tensor([i], device=DEVICE),
                params[i].unsqueeze(0)
            )
            q_vals[i] = q

        idx = int(torch.argmax(q_vals).item())
        return idx, params[idx].cpu().numpy()

def naive_opponent(mask):
    valid = np.flatnonzero(mask)
    if len(valid) == 0: return None
    i = int(np.random.choice(valid))
    fx, fy = np.random.uniform(-0.7, 0.7, 2)
    return np.array([i, fx, fy])

# simulation loop
def simulate(actor_path, critic_path):
    DEVICE = "cuda" if (torch.cuda.is_available()) else "cpu"
    NUM_DISC = 5
    env = AlkkagiEnv(num_agent_discs=NUM_DISC, num_opponent_discs=NUM_DISC)
    obs_dim = env.reset(random=False).size

    actor  = Actor(obs_dim, NUM_DISC).to(DEVICE).eval()
    critic = Critic(obs_dim, NUM_DISC).to(DEVICE).eval()
    actor .load_state_dict(torch.load(actor_path,  map_location=DEVICE))
    critic.load_state_dict(torch.load(critic_path, map_location=DEVICE))
    
    obs = env.reset(random=False).flatten()
    done, tot_r = False, 0.0
    env.render()
    
    while not done:
        mask = env.get_action_mask(0)
        a_idx, a_cont = greedy_action(obs, mask, actor, critic, DEVICE, NUM_DISC)
        obs, r, done, info = env.step(np.array([a_idx, *a_cont]), 0, True)
        tot_r += r
        tot_r -= 1
        obs = obs.flatten()
        time.sleep(1)
        
        if done or NUM_DISC == 1:
            break

        # simple random opponent
        opp = naive_opponent(env.get_action_mask(1))
        if opp is not None:
            obs, r_opp, done, info = env.step(opp, 1, True)
            obs = obs.flatten()
        time.sleep(1)

        if done:
            break

    env.close()

    return tot_r

if __name__ =="__main__":
    print(f"total return: {simulate("ckpt/5/pdqn_actor_50000.pth", f"ckpt/5/pdqn_critic_50000.pth"):.1f}")
    # print(f"total return: {simulate("ckpt/2/pdqn_actor_1000000.pth", f"ckpt/2/pdqn_critic_1000000.pth"):.1f}")
import numpy as np

def hueristic_oppoonent(obs):
    obs = list(zip(*[iter(obs)] * 3))

    opponent_discs = np.array([(info[0], info[1]) for idx, info in enumerate(obs) if idx < len(obs) //2 and info[2] !=2])
    agent_discs = np.array([(info[0], info[1]) for idx, info in enumerate(obs) if idx >= len(obs) //2 and info[2] !=2])

    dists = np.linalg.norm(opponent_discs[:, None] - agent_discs[None, :], axis=2)

    i, j = np.unravel_index(np.argmin(dists), dists.shape)
    direction_vector = agent_discs[j] - opponent_discs[i]

    action = direction_vector / np.linalg.norm(direction_vector)

    return action

hueristic_oppoonent([1,2,0,2,3,0,3,4,0,4,5,1])
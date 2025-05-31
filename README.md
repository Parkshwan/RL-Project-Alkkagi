# RL-Project-Alkkagi

Project for CS377

## Setting environment

```
python -m venv *(env_name)*
source *(env_name)*/bin/activate
pip install -r requirements.txt
```

## Simulation

```
python src/env_simulation.py
```

## Description

### Environment
- Alkkagi_Env(num_agent_discs: int = 1, num_opponent_discs: int = 1)
- action space: [discs' index, x, y]
direction of force will be decided by x and y
- observation space: Tuple((x, y, team) * _(num_agent + num_opponent)_)
x, y is position of disc and team information of disc. Each disc has information of position and team as a environment
you can get current observation with using env._get_obs()

### Test simulation
- Start with 4 of discs each other
- every step, agent's disc go up slightly. after remove all opponent's discs, the simulation will be end.
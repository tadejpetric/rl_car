# /home/tadej/programming/rl_car/venv/lib/python3.13/site-packages/gymnasium/envs/box2d/car_racing.py
# edited line 623

import gymnasium as gym
import math
import numpy as np
import matplotlib.pyplot as plt
#gym.envs.box2d.car_racing.STATE_W = 256
from resized_car_racing import CarRacing2
gym.register(
    id="CarRacing2",
    entry_point=CarRacing2,
)

env = gym.make("CarRacing2", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=True)

observation, info = env.reset()

print(type(observation))
print(observation.shape)

observation, reward, terminated, truncated, info = env.step(np.array([0,1,0]))
print(observation.shape)

print(env)
print(env.observation_space.dtype)
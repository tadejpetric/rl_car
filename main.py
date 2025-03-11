# /home/tadej/programming/rl_car/venv/lib/python3.13/site-packages/gymnasium/envs/box2d/car_racing.py
# edited line 623

import gymnasium as gym
import math
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.envs.box2d.car_racing import MetaData

env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=True)
observation, info = env.reset()

print(type(observation))
for _ in range(10):
    env.step(np.array([0,1,0]))
    env.step(np.array([0,1,0]))
    env.step(np.array([0,1,0]))
    env.step(np.array([0,1,0]))
observation, reward, terminated, truncated, info = env.step(np.array([0,1,0]))
print(observation.shape)
print(info["metastate"])

print(env)
print(env.observation_space.dtype)

# Create a sample RGB array (height=100, width=100)

# Display the image


grass_light = np.array([100,228,100])
grass_dark = np.array([100,202,100])

road = np.array([100,100,100])
toolbar = np.array([0,0,0])
car = np.array([192,15,15])

size_x = 96
size_y = 96
car_x = 48 # 48 +48 = 96 = image width
car_y = 73 # From the top

n_rays = 25

# angle at front of the car = 0
ray_left_angle = -math.pi/2
ray_right_angle = math.pi/2
from dist_utils import get_distance
thetas = np.linspace(ray_left_angle, ray_right_angle, n_rays)

for theta in thetas:
    print(f"theta {theta} : {get_distance(observation, theta)}")
plt.imshow(observation)
plt.axis('off')  # Hide axes
plt.show()
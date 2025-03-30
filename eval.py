import math
from matplotlib import pyplot as plt
import numpy as np
from stable_baselines3 import PPO
import cv2
import gymnasium as gym
from gymnasium.utils.save_video import save_video
from resized_car_racing import CarRacing2


#env = gym.make(
#    "CarRacing-v3",
#    render_mode="rgb_array",
#    lap_complete_percent=0.95,
#    domain_randomize=False,
#    continuous=True
#)

gym.register(
    id="CarRacing2",
    entry_point=CarRacing2,
)

# Create the environment
#env = gym.make(
#    "CarRacing2",
#    render_mode="rgb_array",
#    lap_complete_percent=0.95,
#    domain_randomize=False,
#    continuous=True,
#)

cp_name = "CNN_4_64_up_1000_4096_13_12"
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
def make_env():
    env = gym.make(
        "CarRacing2",
        render_mode="rgb_array",
        lap_complete_percent=0.95,
        domain_randomize=False,
        continuous=True,
    )
    # Apply TimeLimit wrapper (optional, e.g., max 1000 steps)
    env = TimeLimit(env, max_episode_steps=1000)
    return env

vec_env = make_vec_env(make_env, 64)

# (Optional) Stack frames if needed (useful for image-based inputs)
env = VecFrameStack(vec_env, n_stack=4)
model = PPO.load(cp_name, env=env)

ep_len = 1000

renders = []

for ep in range(1):
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    ts = 0
    rews = [0]
    while not done:
        ts += 1
        # Predict an action (set deterministic=True for evaluation)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        renders.append(env.render())

        rews.append(rews[-1] + reward)
        if len(rews) > ep_len:
            break
        total_reward += reward
    print(f"Episode {ep+1}: Total Reward: {total_reward}")
    plt.plot(rews)
    plt.show()

save_video(frames=renders,video_folder="renders",name_prefix=cp_name, fps=30)
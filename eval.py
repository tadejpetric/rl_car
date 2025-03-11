import math
from gymnasium.envs.box2d.car_racing import MetaData
from matplotlib import pyplot as plt
import numpy as np
from stable_baselines3 import PPO
import cv2
from dist_utils import get_distance
import gymnasium as gym

n_rays = 25
ray_left_angle = -math.pi/2
ray_right_angle = math.pi/2
thetas = np.linspace(ray_left_angle, ray_right_angle, n_rays)

def rgb_to_obs(rgb):
    obs = []
    for theta in thetas:
        obs.append(get_distance(rgb, theta))
    return np.array(obs, dtype=np.float32)

def meta_to_obs(meta: MetaData):
    obs = np.array([meta.speed1, meta.speed2, meta.gyro, meta.turning])
    return np.clip(obs, -1, 1)

# Custom wrapper that concatenates info['metastate'] with the observation
class ConcatMetaStateWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # Perform an initial reset to determine the metastate shape.
        obs, info = self.env.reset()
        self.rgb = obs
        meta = info.get("metastate")
        if meta is None:
            raise ValueError("Expected 'metastate' in info but it was not found.")
        meta = np.array(meta)
        
        # Here we assume that the metastate doesn't have defined bounds.
        # You can change -inf and inf to actual limits if available.
        low = -np.ones(n_rays + 4)
        high = np.ones(n_rays + 4)

        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, **kwargs):
        rgb, info = self.env.reset(**kwargs)

        self.rgb = rgb 
        obs1 = rgb_to_obs(rgb)
        meta = info.get("metastate")
        if meta is None:
            raise ValueError("Expected 'metastate' in info but it was not found.")
        obs2 = meta_to_obs(meta)
        new_obs = np.concatenate([obs1, obs2])
        return new_obs, info

    def step(self, action):
        rgb, reward, terminated, truncated, info = self.env.step(action)
        self.rgb = rgb
        obs1 = rgb_to_obs(rgb)
        meta = info.get("metastate")
        if meta is None:
            raise ValueError("Expected 'metastate' in info but it was not found.")
        obs2 = meta_to_obs(meta)
        new_obs = np.concatenate([obs1, obs2])
        return new_obs, reward, terminated, truncated, info

# Replace 'YourCustomEnv-v0' with your environment id or instance.

from gymnasium.wrappers import TimeLimit

def gym_make():
    env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=True)
    return ConcatMetaStateWrapper(env)

# Define video parameters
output_filename = "output_video.mp4"
frame_width = 96
frame_height = 96
fps = 30  # Frames per second
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec

# Create VideoWriter object
out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

# Example: Generating random 96x96 RGB frames
num_frames = 100  # Number of frames in the video
#env = gym_make()

env = gym.make(
    "CarRacing-v3",
    render_mode="rgb_array",
    lap_complete_percent=0.95,
    domain_randomize=False,
    continuous=True
)

model = PPO.load("ppo_carracing_custom_cnn5", env=env)

ep_len = 500


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
        
        out.write(obs)

        rews.append(rews[-1] + reward)
        if len(rews) > ep_len:
            break
        total_reward += reward
    print(f"Episode {ep+1}: Total Reward: {total_reward}")
    plt.plot(rews)
    plt.show()

# Release the video writer
out.release()


import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from dist_utils import get_distance
import math
from gymnasium.envs.box2d.car_racing import MetaData


n_rays = 25
ray_left_angle = -math.pi / 2
ray_right_angle = math.pi / 2
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

        obs1 = rgb_to_obs(rgb)
        meta = info.get("metastate")
        if meta is None:
            raise ValueError("Expected 'metastate' in info but it was not found.")
        obs2 = meta_to_obs(meta)
        new_obs = np.concatenate([obs1, obs2])
        return new_obs, info

    def step(self, action):
        rgb, reward, terminated, truncated, info = self.env.step(action)
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
    env = gym.make(
        "CarRacing-v3",
        render_mode="rgb_array",
        lap_complete_percent=0.95,
        domain_randomize=False,
        continuous=True,
    )
    env = TimeLimit(env, 1000)
    return ConcatMetaStateWrapper(env)


# Optionally, create a vectorized environment for parallelization.
vec_env = make_vec_env(gym_make, n_envs=1)

# Create a PPO model using an MLP policy.
import os

if os.path.exists("ppo_navigate_model2.zip"):
    model = PPO.load(
        "ppo_navigate_model2",
        env=vec_env,
        verbose=1,
        n_steps=256,
        learning_rate=0.0005,
        batch_size=16,
    )
    print("loaded")
else:
    model = PPO("MlpPolicy", vec_env, n_steps=256, verbose=1)
    print("inited")


# Train the model; adjust total_timesteps as needed.
max_timesteps_per_episode = 1000
episodes = 1
model.learn(total_timesteps=2000, progress_bar=True)

# Save the trained model.
model.save("ppo_navigate_model3")

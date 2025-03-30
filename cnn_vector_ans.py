import os
import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from resized_car_racing import CarRacing2

gym.register(
    id="CarRacing2",
    entry_point=CarRacing2,
)

env = gym.make("CarRacing2", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=True)

# Define a custom CNN feature extractor
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # print(observation_space.shape) # -> (3, 224, 224)

        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            # Conv2d expects input shape: (batch_size, channels, height, width)
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Do a dummy forward pass to determine the output size of the CNN
        with th.no_grad():
            sample_obs = observation_space.sample()
            sample_obs = th.as_tensor(sample_obs, dtype=th.float32).unsqueeze(0)
            n_flatten = self.cnn(sample_obs).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations):
        observations = observations.float()
        cnn_output = self.cnn(observations)
        return self.linear(cnn_output)


# Create the CarRacing-v3 environment
env = gym.make(
    "CarRacing2",
    render_mode="rgb_array",
    lap_complete_percent=0.95,
    domain_randomize=False,
    continuous=True,
)

# Define policy keyword arguments to use the custom CNN
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=256),
)

from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines3.common.env_util import make_vec_env

# Initialize the PPO agent with the custom CNN policy.
num_envs = 64
num_stacks = 4
episode_steps = 1000
n_steps = 4096

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

# Number of parallel environments
vec_env = make_vec_env(make_env, num_envs)

# (Optional) Stack frames if needed (useful for image-based inputs)
vec_env = VecFrameStack(vec_env, n_stack=num_stacks)

old_name = f"CNN_4_64_up_1000_4096_12_11"
new_name = f"CNN_{num_stacks}_{num_envs}_up_{episode_steps}_{n_steps}_13_12"
if os.path.exists(f"{old_name}.zip"):
    model = PPO.load(
        old_name,
        policy="CnnPolicy",
        env=TimeLimit(env, episode_steps),
        verbose=1,
        n_steps=n_steps,
        batch_size=512,
        clip_range=0.2,
        learning_rate=1e-4,
        tensorboard_log="./tensorboard/",
        device="cuda"
    )
    print(model.observation_space)
    print("loaded")
else:
    model = PPO(
        "CnnPolicy",
        env=TimeLimit(env, episode_steps),
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=n_steps,
        batch_size=512,
        clip_range=0.2,
        learning_rate=3e-4,
        tensorboard_log="./tensorboard/",
        device="cuda",  # Use CPU for training
    )
    print("inited")

# Train the agent
# log name is <network>_<time limit>_<n steps>_<run id>_<pretrain id>
model.learn(total_timesteps=256 * 1024, progress_bar=True, tb_log_name=new_name)

# Optionally, save the model
model.save(new_name)
print(new_name)
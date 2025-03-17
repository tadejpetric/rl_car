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

# Initialize the PPO agent with the custom CNN policy.

old_name = "xxCNN_upscaled_512_256_1_0"
new_name = "CNN_upscaled_1000_1000_3_0"
if os.path.exists(f"{old_name}.zip"):
    model = PPO.load(
        old_name,
        policy="CnnPolicy",
        env=TimeLimit(env, 1000),
        learning_rate=1e-4,
        n_steps=1000,
        tensorboard_log="./tensorboard/",
        device="cpu"
    )
    print("loaded")
else:
    model = PPO(
        "CnnPolicy",
        env=TimeLimit(env, 1000),
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=1000,
        learning_rate=3e-4,
        tensorboard_log="./tensorboard/",
        device="cpu",  # Use CPU for training
    )
    print("inited")

# Train the agent
# log name is <network>_<time limit>_<n steps>_<run id>_<pretrain id>
model.learn(total_timesteps=256 * 128, progress_bar=True, tb_log_name=new_name)

# Optionally, save the model
model.save(new_name)
import os
import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Define a custom CNN feature extractor
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # The observation space for CarRacing-v3 is expected to be either (96, 96, 3) or (3, 96, 96)
        # If it's (96, 96, 3), then the channel dimension is at index 2, otherwise at index 0.
        # Here we assume that if observation_space.shape[2] exists, then it's HxWxC.
        if len(observation_space.shape) == 3 and observation_space.shape[0] == 3:
            n_input_channels = observation_space.shape[0]
        else:
            n_input_channels = observation_space.shape[2]
        
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
            # Check if we need to permute: if the observation is HxWxC (i.e., shape[0] != 3), permute it.
            if sample_obs.ndim == 4 and sample_obs.shape[1] != 3:
                sample_obs = sample_obs.permute(0, 3, 1, 2)
            n_flatten = self.cnn(sample_obs).shape[1]
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations):
        # observations might be in (B, H, W, C) or (B, C, H, W)
        # Check if the second dimension is not 3; if so, assume the format is HxWxC and permute.
        if observations.ndim == 4 and observations.shape[1] != 3:
            observations = observations.permute(0, 3, 1, 2)
        observations = observations.float()
        cnn_output = self.cnn(observations)
        return self.linear(cnn_output)

# Create the CarRacing-v3 environment
env = gym.make(
    "CarRacing-v3",
    render_mode="rgb_array",
    lap_complete_percent=0.95,
    domain_randomize=False,
    continuous=True
)

# Define policy keyword arguments to use the custom CNN
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=256),
)

from gymnasium.wrappers import TimeLimit
# Initialize the PPO agent with the custom CNN policy.
if os.path.exists("ppo_carracing_custom_cnn4.zip"):
    model = PPO.load(
    "ppo_carracing_custom_cnn4",
    env=env,
    n_steps=1000,
    tensorboard_log="./tensorboard/"
    )
    print("loaded")
else:
    model = PPO(
        "CnnPolicy",
        env=TimeLimit(env, 500),
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=512,
        learning_rate=3e-4,
        device="cpu"  # Use CPU for training
    )
    print("inited")

# Train the agent
model.learn(total_timesteps=100_000, progress_bar=True)

# Optionally, save the model
model.save("ppo_carracing_custom_cnn5")

# To load the trained model later:
# model = PPO.load("ppo_carracing_custom_cnn", env=env)

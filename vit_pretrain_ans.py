import os
import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
import timm  # Make sure you have timm installed: pip install timm

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from resized_car_racing import CarRacing2

# Register the custom environment
gym.register(
    id="CarRacing2",
    entry_point=CarRacing2,
)

# Create the environment
env = gym.make(
    "CarRacing2",
    render_mode="rgb_array",
    lap_complete_percent=0.95,
    domain_randomize=False,
    continuous=True,
)

# Define a custom ViT feature extractor using a pretrained ViT-B/16 model.
class CustomViT(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(CustomViT, self).__init__(observation_space, features_dim)
        # Load a pretrained ViT model.
        # Here we use 'vit_base_patch16_224', which expects 224x224 inputs.
        # Alternatively, you can use 'vit_base_patch32_224' for a 32px patch size.
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        # Remove the classification head to obtain the raw features.
        self.vit.head = nn.Identity()
        # The ViT outputs 768-dimensional features. We add an adapter to map these to the desired feature_dim.
    
    def forward(self, observations):
        # Convert observations to float and scale from [0, 255] to [0, 1].
        observations = observations.float() / 255.0
        # Normalize using ImageNet statistics.
        mean = th.tensor([0.485, 0.456, 0.406], device=observations.device).view(1, -1, 1, 1)
        std = th.tensor([0.229, 0.224, 0.225], device=observations.device).view(1, -1, 1, 1)
        observations = (observations - mean) / std
        # Pass the normalized image through the ViT to get a 768-dim representation.
        vit_features = self.vit(observations)  # Expected shape: (batch, 768)
        # Use the adapter to produce the final feature vector.
        return vit_features

# Recreate the environment (if needed)
env = gym.make(
    "CarRacing2",
    render_mode="rgb_array",
    lap_complete_percent=0.95,
    domain_randomize=False,
    continuous=True,
)

# Specify policy keyword arguments to use the custom ViT feature extractor.
policy_kwargs = dict(
    features_extractor_class=CustomViT,
    features_extractor_kwargs=dict(features_dim=768),
)

from gymnasium.wrappers import TimeLimit

# Initialize the PPO agent with the custom ViT policy.
if os.path.exists("ppo_carracing_custom_vit.zip"):
    model = PPO.load(
        "ppo_carracing_custom_vit",
        env=TimeLimit(env, 512),
        n_steps=512,
        tensorboard_log="./tensorboard/",
    )
    print("loaded")
else:
    model = PPO(
        "CnnPolicy",  # The policy type remains CnnPolicy, but the feature extractor is overridden.
        env=TimeLimit(env, 256),
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=256,
        learning_rate=3e-4,
        tensorboard_log="./tensorboard/",
        device="cuda",  # Use CPU for training
    )
    print("inited")

# Train the agent
model.learn(total_timesteps=256 * 100, progress_bar=True, tb_log_name="ViT_run")

# Optionally, save the model
model.save("ppo_carracing_custom_vit")

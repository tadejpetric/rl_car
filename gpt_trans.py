import os
import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium.wrappers import TimeLimit

# Define a custom Transformer feature extractor
class CustomTransformer(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256, patch_size=16, num_layers=4, num_heads=8, dropout=0.01):
        super(CustomTransformer, self).__init__(observation_space, features_dim)
        # Determine input shape and channels.
        shape = observation_space.shape
        if len(shape) == 3:
            # Check if the observation is in (C, H, W) or (H, W, C)
            if shape[0] == 3:
                channels, height, width = shape
            else:
                height, width, channels = shape
        else:
            raise NotImplementedError("Observation space shape not supported")

        self.patch_size = patch_size
        self.embed_dim = features_dim  # Transformer embedding dimension

        # Calculate number of patches per image. For example, with a 96x96 image and patch_size 16, you get (96/16)^2 = 36 patches.
        self.num_patches = (height // patch_size) * (width // patch_size)

        # Patch embedding: use a Conv2d with kernel and stride equal to patch_size.
        self.patch_embed = nn.Conv2d(channels, self.embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Learnable positional embedding for each patch.
        self.pos_embedding = nn.Parameter(th.zeros(1, self.num_patches, self.embed_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        self.dropout = nn.Dropout(dropout)

        # Define the Transformer encoder: using PyTorch's built-in transformer encoder layers.
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # MLP head to map the aggregated transformer output to the desired features dimension.
        self.mlp_head = nn.Sequential(
            nn.Linear(self.embed_dim, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations):
        # observations can be (B, H, W, C) or (B, C, H, W). Ensure they are in (B, C, H, W)
        if observations.ndim == 4:
            if observations.shape[1] != 3:  # assume (B, H, W, C)
                observations = observations.permute(0, 3, 1, 2)
        observations = observations.float()
        
        # Apply patch embedding: resulting shape (B, embed_dim, H/patch_size, W/patch_size)
        x = self.patch_embed(observations)
        # Flatten spatial dimensions to obtain patch tokens: (B, embed_dim, num_patches)
        x = x.flatten(2)
        # Transpose to shape (B, num_patches, embed_dim)
        x = x.transpose(1, 2)
        
        # Add positional embeddings and apply dropout.
        x = x + self.pos_embedding
        x = self.dropout(x)
        
        # Transformer expects (sequence_length, batch_size, embed_dim)
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)
        # Transpose back to (B, num_patches, embed_dim)
        x = x.transpose(0, 1)
        
        # Aggregate the patch tokens (e.g., by averaging over the sequence dimension)
        x = th.mean(x, dim=1)
        return self.mlp_head(x)

# Create the CarRacing-v3 environment
env = gym.make(
    "CarRacing-v3",
    render_mode="rgb_array",
    lap_complete_percent=0.95,
    domain_randomize=False,
    continuous=True
)

# Define policy keyword arguments to use the custom Transformer feature extractor
policy_kwargs = dict(
    features_extractor_class=CustomTransformer,
    features_extractor_kwargs=dict(features_dim=256, patch_size=8, num_layers=4, num_heads=8, dropout=0.01),
)

# Initialize the PPO agent with the custom Transformer policy.
model_path = "ppo_carracing_transformer.zip"
if os.path.exists(model_path):
    model = PPO.load(
        model_path,
        env=env,
        n_steps=1000,
        tensorboard_log="./tensorboard/"
    )
    print("Loaded existing model")
else:
    model = PPO(
        "CnnPolicy",  # The policy type remains "CnnPolicy", but our custom extractor will be used.
        env=TimeLimit(env, 256),
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=256,
        learning_rate=3e-4,
        tensorboard_log="./tensorboard/",
        device="cpu"  # Use CPU for training
    )
    print("Initialized new model")

# Train the agent
model.learn(total_timesteps=256*100, progress_bar=True)

# Optionally, save the model
model.save("ppo_carracing_transformer_trained")

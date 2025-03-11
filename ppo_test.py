import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import numpy as np

# Replace 'YourCustomEnv-v0' with your actual environment id or your custom environment instance.
# For example, if you have a custom environment class, you could do:
# env = YourCustomEnv(...)

env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=True)


# If your environment returns a 3D np array and your policy expects a flat vector,
# you can wrap your env to flatten observations. Here’s a simple example wrapper:
class FlattenObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        original_space = self.observation_space
        # Flatten the original observation space into a 1D Box.
        low = original_space.low.flatten()
        high = original_space.high.flatten()
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=original_space.dtype)
    
    def observation(self, observation):
        return observation.flatten()

# Uncomment the following line if you need to flatten your observations:
# env = FlattenObservation(env)

# Optionally, create a vectorized environment (helps with parallelization).
vec_env = make_vec_env(lambda: env, n_envs=1)

# Create the PPO model with a multi-layer perceptron (MLP) policy.
model = PPO("MlpPolicy", vec_env, verbose=1)

# Train the model. Adjust total_timesteps as needed.
model.learn(total_timesteps=100_000)

# Save the trained model.
model.save("ppo_navigate_model")

# To demonstrate evaluation, reload the model and run a few episodes.
model = PPO.load("ppo_navigate_model")

episodes = 5
for ep in range(episodes):
    obs, _ = env.reset()  # gymnasium reset returns (observation, info)
    done = False
    total_reward = 0.0
    while not done:
        # Use the model to predict an action (deterministic for evaluation)
        action, _ = model.predict(obs, deterministic=True)
        # Step the environment. Note: Gymnasium now returns (obs, reward, terminated, truncated, info)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
    print(f"Episode {ep+1}: Total Reward: {total_reward}")

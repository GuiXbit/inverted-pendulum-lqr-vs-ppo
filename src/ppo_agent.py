import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
import os

env = gym.make("CartPole-v1")

MODEL_PATH = "results/ppo_cartpole"

if os.path.exists(MODEL_PATH + ".zip"):
    print("Loading existing model...")
    model = PPO.load(MODEL_PATH)
else:
    print("Training new model...")
    env = gym.make("CartPole-v1")
    model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=2048, verbose=1)
    model.learn(total_timesteps=100_000)
    model.save(MODEL_PATH)
    env.close()

eval_env = gym.make("CartPole-v1", render_mode="human")

episode_rewards = []
episode_control_efforts = []
episode_state_norms = []

for _ in range(2):
    obs, _ = eval_env.reset()
    current_rewards = []
    current_efforts = []
    current_norms = []
    terminated = truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(obs)
        current_efforts.append(abs(float(action)))
        current_norms.append(np.linalg.norm(obs))
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        current_rewards.append(reward)

    episode_rewards.append(sum(current_rewards))
    episode_control_efforts.append(np.mean(current_efforts))
    episode_state_norms.append(np.mean(current_norms))

eval_env.close()

print(f"Avg episode reward: {np.mean(episode_rewards):.2f}")
print(f"Avg control effort: {np.mean(episode_control_efforts):.2f}")
print(f"Avg state norm:     {np.mean(episode_state_norms):.2f}")
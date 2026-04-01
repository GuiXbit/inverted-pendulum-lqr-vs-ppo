import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
import os

MODEL_PATH = "results/ppo_cartpole"

def train_or_load_ppo():
    if os.path.exists(MODEL_PATH + ".zip"):
        print("Loading existing model...")
        return PPO.load(MODEL_PATH)
    print("Training new model...")
    env = gym.make("CartPole-v1")
    model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=2048, verbose=1)
    model.learn(total_timesteps=100_000)
    model.save(MODEL_PATH)
    env.close()
    return model

def run_ppo(n_episodes=20, render=False, noise_std=0):
    model = train_or_load_ppo()
    render_mode = "human" if render else None
    eval_env = gym.make("CartPole-v1", render_mode=render_mode)

    episode_rewards, episode_efforts, episode_norms = [], [], []

    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        current_rewards, current_efforts, current_norms = [], [], []
        terminated = truncated = False
        #adicionar ruido
        

        while not (terminated or truncated):
            obs_noisy = obs.copy()
            obs_noisy[2] += np.random.normal(0, noise_std)
            action, _ = model.predict(obs_noisy)
            current_efforts.append(abs(float(action)))
            current_norms.append(np.linalg.norm(obs))
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            current_rewards.append(reward)
        episode_rewards.append(sum(current_rewards))
        episode_efforts.append(np.mean(current_efforts))
        episode_norms.append(np.mean(current_norms))

    eval_env.close()
    return episode_rewards, episode_efforts, episode_norms
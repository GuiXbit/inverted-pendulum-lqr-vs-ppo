import numpy as np
from scipy.linalg import solve_continuous_are
import gymnasium as gym

def build_lqr():
    M, m, l, g, I = 1.0, 0.1, 0.5, 9.8, 0.0
    p = I*(M+m) + M*m*l**2

    A = np.array([
        [0, 1,             0,          0],
        [0, 0,  m**2*g*l**2/p,         0],
        [0, 0,             0,          1],
        [0, 0,  m*g*l*(M+m)/p,         0]
    ])
    B = np.array([[0], [(I+m*l**2)/p], [0], [m*l/p]])
    Q = np.diag([1, 1, 1, 1])
    R = np.array([[1]])
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    return K


def run_lqr(n_steps=1000, render=False,noise_std=0):
    K = build_lqr()
    render_mode = "human" if render else None
    env = gym.make("CartPole-v1", render_mode=render_mode)
    obs, _ = env.reset()

    episode_rewards, episode_efforts, episode_norms = [], [], []
    current_rewards, current_efforts, current_norms = [], [], []

    for _ in range(n_steps):
        obs_noisy = obs.copy()
        obs_noisy[2] += np.random.normal(0, noise_std)
        u = (-K @ obs_noisy).item()
        action = 0 if u > 0 else 1
        current_efforts.append(abs(u))
        current_norms.append(np.linalg.norm(obs))
        obs, reward, terminated, truncated, _ = env.step(action)
        current_rewards.append(reward)
        if terminated or truncated:
            episode_rewards.append(sum(current_rewards))
            episode_efforts.append(np.mean(current_efforts))
            episode_norms.append(np.mean(current_norms))
            current_rewards, current_efforts, current_norms = [], [], []
            obs, _ = env.reset()

    env.close()
    return episode_rewards, episode_efforts, episode_norms
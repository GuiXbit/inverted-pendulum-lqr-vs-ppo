from src.lqr_controller import run_lqr
from src.ppo_agent import run_ppo
import numpy as np

def print_results(label, rewards, efforts, norms):
    print(f"{label} | Avg reward: {np.mean(rewards):.2f} | Avg effort: {np.mean(efforts):.2f} | Avg norm: {np.mean(norms):.2f}")

if __name__ == "__main__":
    print("=== No noise ===")
    print_results("LQR", *run_lqr(n_steps=1000))
    print_results("PPO", *run_ppo(n_episodes=20))

    print("\n=== Noise std=0.1 ===")
    print_results("LQR", *run_lqr(n_steps=1000, noise_std=0.1))
    print_results("PPO", *run_ppo(n_episodes=20, noise_std=0.1))
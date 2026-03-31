from src.lqr_controller import run_lqr
from src.ppo_agent import run_ppo

if __name__ == "__main__":
    print("Running LQR...")
    lqr_rewards, lqr_efforts, lqr_norms = run_lqr(n_steps=1000)

    print("Running PPO...")
    ppo_rewards, ppo_efforts, ppo_norms = run_ppo(n_episodes=20)

    print("\n--- Results ---")
    print(f"LQR | Avg reward: {sum(lqr_rewards)/len(lqr_rewards):.2f} | Avg effort: {sum(lqr_efforts)/len(lqr_efforts):.2f} | Avg norm: {sum(lqr_norms)/len(lqr_norms):.2f}")
    print(f"PPO | Avg reward: {sum(ppo_rewards)/len(ppo_rewards):.2f} | Avg effort: {sum(ppo_efforts)/len(ppo_efforts):.2f} | Avg norm: {sum(ppo_norms)/len(ppo_norms):.2f}")
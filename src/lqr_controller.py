import numpy as np
from scipy.linalg import solve_continuous_are
import gymnasium as gym

# cart pole parameters 
M = 1.0    
m = 0.1    
l = 0.5   
g = 9.8
I = 0.0  

p=I*(M+m)+M*m*l**2
A = np.array([
    [0,           1,              0,          0],
    [0,           0,   m**2*g*l**2/p,         0],
    [0,           0,              0,          1],
    [0,           0,   m*g*l*(M+m)/p,         0]
])
 
B = np.array([
    [0],
    [(I + m*l**2) / p],
    [0],
    [m*l / p]
])

#weight matrices
Q = np.diag([1, 1, 100, 10]) 

R = np.array([[1]])

P = solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P
print("K Gain:", K)


env = gym.make("CartPole-v1", render_mode="human")
obs, _ = env.reset()

episode_rewards = []
episode_control_efforts = []
episode_state_norms = []

current_rewards = []
current_efforts = []
current_norms = []

for _ in range(1000):
    u = (-K @ obs).item()
    action = 0 if u > 0 else 1

    current_efforts.append(abs(u))
    current_norms.append(np.linalg.norm(obs))

    obs, reward, terminated, truncated, _ = env.step(action)
    current_rewards.append(reward)

    if terminated or truncated:
        episode_rewards.append(sum(current_rewards))
        episode_control_efforts.append(np.mean(current_efforts))
        episode_state_norms.append(np.mean(current_norms))
        current_rewards, current_efforts, current_norms = [], [], []
        obs, _ = env.reset()

env.close()

print(f"Avg episode reward: {np.mean(episode_rewards):.2f}")
print(f"Avg control effort: {np.mean(episode_control_efforts):.2f}")
print(f"Avg state norm:     {np.mean(episode_state_norms):.2f}")
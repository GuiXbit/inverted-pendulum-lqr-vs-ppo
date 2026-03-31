# Inverted Pendulum — LQR vs PPO

Comparison between a classical optimal controller (LQR) and a reinforcement learning agent (PPO) on the CartPole-v1 environment.

This project was developed as part of a personal study initiative inspired by the course *Decision Making for Mobile Robots using Reinforcement Learning* at ICMC – USP.

---

## Problem

The CartPole task consists of balancing an inverted pendulum mounted on a cart by applying horizontal forces. The system state is described by four variables:

| Variable | Description |
|---|---|
| `x` | Cart position |
| `x_dot` | Cart velocity |
| `theta` | Pole angle |
| `theta_dot` | Pole angular velocity |

The goal is to keep the pole upright for as long as possible.

---

## Approaches

### LQR (Linear Quadratic Regulator)
Classical optimal control method. The system is linearized around the equilibrium point and the optimal gain matrix **K** is computed by solving the Algebraic Riccati Equation.

### PPO (Proximal Policy Optimization)
Model-free reinforcement learning algorithm. The agent learns a policy purely through interaction with the environment, with no knowledge of the system dynamics.

---

## Results

| Method | Avg Reward | Avg Control Effort | Avg State Norm |
|---|---|---|---|
| LQR | 500.00 | 1.18 | 0.44 |
| PPO | 500.00 | 0.50 | 0.23 |

Both controllers achieve maximum reward. PPO produces smoother control actions and keeps the system closer to equilibrium, at the cost of requiring 100k timesteps of training. 

**WIP:**
- Comparison plots (reward, control effort, state norm)
- Robustness test under external disturbances

---

## Setup

```bash
git clone https://github.com/GuiXbit/inverted-pendulum-lqr-vs-ppo.git
cd inverted-pendulum-lqr-vs-ppo
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

---

## Project Structure

```
inverted-pendulum-lqr-vs-ppo/
├── main.py
├── requirements.txt
├── src/
│   ├── lqr_controller.py
│   └── ppo_agent.py
└── results/
    └── plots/
```

---

## Author

Gui Mendonça — M.Sc. student in Electrical Engineering (Dynamic Systems & Control), EESC–USP.

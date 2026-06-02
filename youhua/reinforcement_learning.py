# -*- coding: utf-8 -*-
"""
reinforcement_learning.py
-------------------------
Self-contained deep reinforcement learning optimizer for continuous CST design
variables. It uses a small DDPG-style actor-critic implemented with PyTorch.

The public entry point is run_rl(...), matching run_sa/run_ga/run_pso:
    result = run_rl(objective_func, bounds, ...)

objective_func(x) must return a dict containing at least:
    fitness, cl, cd, t_max, penalty, feasible
"""

from collections import deque
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_arrays(bounds):
    low = np.array([b[0] for b in bounds], dtype=np.float32)
    high = np.array([b[1] for b in bounds], dtype=np.float32)
    width = high - low
    return low, high, width


def _clip_x(x, low, high):
    return np.clip(np.asarray(x, dtype=np.float32), low, high)


def _normalize_x(x, low, high):
    return 2.0 * (np.asarray(x, dtype=np.float32) - low) / (high - low) - 1.0


def _make_state(x, result, low, high):
    x_norm = _normalize_x(x, low, high)
    extra = np.array(
        [
            float(result.get("cl", 0.0)),
            float(result.get("cd", 0.0)) * 1000.0,
            float(result.get("t_max", 0.0)) * 10.0,
            np.log1p(min(float(result.get("penalty", 0.0)), 1e6)),
            1.0 if result.get("feasible", False) else 0.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([x_norm, extra]).astype(np.float32)


def _objective_score(result):
    return 1000.0 * float(result.get("fitness", np.inf))


def _calc_reward(prev_result, next_result, best_score_before):
    prev_score = _objective_score(prev_result)
    next_score = _objective_score(next_result)

    # 基于相对改善比例的奖励（而非绝对差值，避免大数值淹没信号）
    if abs(prev_score) > 1e-8:
        improvement_ratio = (prev_score - next_score) / abs(prev_score)
    else:
        improvement_ratio = 0.0
    reward = improvement_ratio * 100.0

    # 可行性奖励
    if next_result.get("feasible", False):
        reward += 0.1
    else:
        reward -= 0.1

    # 全局最优改善：给予更大幅度的奖励
    if next_score < best_score_before:
        delta_ratio = (best_score_before - next_score) / max(abs(best_score_before), 1e-8)
        reward += 2.0 * delta_ratio * 100.0

    return float(np.clip(reward, -10.0, 10.0))


class ReplayBuffer:
    def __init__(self, capacity):
        self.data = deque(maxlen=capacity)

    def __len__(self):
        return len(self.data)

    def add(self, state, action, reward, next_state, done):
        self.data.append((state, action, reward, next_state, done))

    def sample(self, batch_size, device):
        batch = random.sample(self.data, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.tensor(np.asarray(states), dtype=torch.float32, device=device),
            torch.tensor(np.asarray(actions), dtype=torch.float32, device=device),
            torch.tensor(np.asarray(rewards), dtype=torch.float32, device=device).unsqueeze(1),
            torch.tensor(np.asarray(next_states), dtype=torch.float32, device=device),
            torch.tensor(np.asarray(dones), dtype=torch.float32, device=device).unsqueeze(1),
        )


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=(128, 128)):
        super().__init__()
        h1, h2 = hidden_sizes
        self.net = nn.Sequential(
            nn.Linear(state_dim, h1),
            nn.LayerNorm(h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.LayerNorm(h2),
            nn.ReLU(),
            nn.Linear(h2, action_dim),
            nn.Tanh(),
        )

    def forward(self, state):
        return self.net(state)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=(128, 128)):
        super().__init__()
        h1, h2 = hidden_sizes
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, h1),
            nn.LayerNorm(h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.LayerNorm(h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))


def _soft_update(target, source, tau):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)


def _select_initial_x(ep, low, high, width, best_x, center_x, rng):
    if ep == 0:
        return center_x.copy()
    if best_x is not None and rng.random() < 0.75:
        return _clip_x(best_x + rng.normal(0.0, 0.06, size=len(low)) * width, low, high)
    return rng.uniform(low, high).astype(np.float32)


def run_rl(
    objective_func,         # 目标函数，接收设计变量 x，返回包含 fitness/cl/cd 等的字典
    bounds,                 # 各设计变量的取值范围列表，如 [(low1, high1), (low2, high2), ...]
    episodes=120,           # 训练总轮数（episode 数量）
    steps_per_episode=40,   # 每个 episode 内的最大步数
    step_ratio=0.08,        # 每步动作幅度占搜索空间宽度的比例（会随 episode 线性衰减）
    gamma=0.98,             # 折扣因子（discount factor），控制未来奖励的权重
    tau=0.01,               # 目标网络软更新系数（soft update），越小更新越慢越稳定
    actor_lr=1e-4,          # Actor 网络的学习率
    critic_lr=3e-4,         # Critic 网络的学习率
    batch_size=128,         # 经验回放采样的批量大小
    replay_capacity=50000,  # 经验回放缓冲区的最大容量
    warmup_steps=500,       # 随机探索的预热步数，之后才开始使用策略
    exploration_noise=0.35, # 初始探索噪声标准差（叠加在动作上的高斯噪声）
    min_noise=0.04,         # 探索噪声的下限值
    noise_decay=0.995,      # 每个 episode 后噪声的衰减系数（乘以此值）
    hidden_sizes=(128, 128),# Actor/Critic 隐藏层的神经元数量
    random_seed=42,         # 随机种子，保证结果可复现
    record_indices=(0, 1),  # 用于记录过程的变量索引（如选第0、1个设计变量绘图）
    device=None,            # 计算设备，默认自动选择 CUDA 或 CPU
    updates_per_step=2,     # 每步从 replay buffer 采样并更新网络的次数
    patience=25,            # 连续多少轮无改善时触发探索重启
):
    """
    Run a DDPG-style continuous-action optimizer.

    State: normalized CST parameters plus predicted cl/cd/t_max/penalty/feasible.
    Action: bounded incremental update for the CST parameters.
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    low, high, width = _as_arrays(bounds)
    dim = len(low)
    center_x = ((low + high) * 0.5).astype(np.float32)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    state_dim = dim + 5
    action_dim = dim

    actor = Actor(state_dim, action_dim, hidden_sizes=hidden_sizes).to(device)
    critic = Critic(state_dim, action_dim, hidden_sizes=hidden_sizes).to(device)
    target_actor = Actor(state_dim, action_dim, hidden_sizes=hidden_sizes).to(device)
    target_critic = Critic(state_dim, action_dim, hidden_sizes=hidden_sizes).to(device)
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)
    replay = ReplayBuffer(replay_capacity)

    rng = np.random.default_rng(random_seed)

    best_x = None
    best_result = None
    best_score = np.inf
    history = []
    total_steps = 0
    noise_std = exploration_noise
    i1, i2 = record_indices

    # 自适应探索重启：跟踪连续无改善的轮数
    episodes_since_improvement = 0
    min_step_ratio = step_ratio * 0.25  # 步长衰减下限

    for ep in range(episodes):
        # 步长随 episode 线性衰减（前期大步探索，后期精细调优）
        ep_ratio = ep / max(episodes - 1, 1)
        current_step_ratio = step_ratio * (1.0 - ep_ratio * 0.5) + min_step_ratio * (ep_ratio * 0.5)

        # 自适应探索重启：连续 patience 轮无改善时放大噪声并增加随机重启概率
        force_random_start = episodes_since_improvement >= patience
        if force_random_start:
            noise_std = max(exploration_noise * 0.5, noise_std)
            x = rng.uniform(low, high).astype(np.float32)
        else:
            x = _select_initial_x(ep, low, high, width, best_x, center_x, rng)

        best_score_before_ep = best_score

        result = objective_func(x)

        score = _objective_score(result)
        if score < best_score:
            best_score = score
            best_x = x.copy()
            best_result = result.copy()

        state = _make_state(x, result, low, high)

        for step in range(steps_per_episode):
            total_steps += 1

            if total_steps <= warmup_steps:
                action = rng.uniform(-1.0, 1.0, size=dim).astype(np.float32)
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    action = actor(state_tensor).cpu().numpy()[0]
                action += rng.normal(0.0, noise_std, size=dim).astype(np.float32)
                action = np.clip(action, -1.0, 1.0).astype(np.float32)

            x_next = _clip_x(x + action * current_step_ratio * width, low, high)
            next_result = objective_func(x_next)
            next_state = _make_state(x_next, next_result, low, high)

            reward = _calc_reward(result, next_result, best_score)
            done = float(step == steps_per_episode - 1)
            replay.add(state, action, reward, next_state, done)

            next_score = _objective_score(next_result)
            if next_score < best_score:
                best_score = next_score
                best_x = x_next.copy()
                best_result = next_result.copy()

            if len(replay) >= batch_size:
                for _ in range(updates_per_step):
                    s, a, r, ns, d = replay.sample(batch_size, device)

                    with torch.no_grad():
                        next_action = target_actor(ns)
                        target_q = target_critic(ns, next_action)
                        y = r + gamma * (1.0 - d) * target_q

                    q = critic(s, a)
                    critic_loss = F.mse_loss(q, y)
                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    critic_optimizer.step()

                    actor_loss = -critic(s, actor(s)).mean()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    _soft_update(target_actor, actor, tau)
                    _soft_update(target_critic, critic, tau)

            x = x_next
            result = next_result
            state = next_state

            history.append(
                {
                    "iter": total_steps,
                    "episode": ep + 1,
                    "step": step + 1,
                    "p1": float(x[i1]),
                    "p2": float(x[i2]),
                    "fitness": float(result["fitness"]),
                    "best_fitness": float(best_result["fitness"]),
                    "cl": float(result["cl"]),
                    "cd": float(result["cd"]),
                    "t_max": float(result["t_max"]),
                    "best_cd": float(best_result["cd"]),
                    "reward": float(reward),
                    "feasible": int(result["feasible"]),
                }
            )

        # 更新无改善计数器
        if best_score < best_score_before_ep:
            episodes_since_improvement = 0
        else:
            episodes_since_improvement += 1

        noise_std = max(min_noise, noise_std * noise_decay)

    if best_x is None or best_result is None:
        raise RuntimeError("RL optimizer did not evaluate any candidate.")

    return {
        "algorithm": "RL",
        "best_x": np.asarray(best_x, dtype=float),
        "best_result": best_result,
        "history": history,
    }

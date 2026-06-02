# -*- coding: utf-8 -*-
"""
particle_swarm.py
-----------------
本文件实现粒子群算法（Particle Swarm Optimization, PSO）。

适用场景：
- 连续优化问题
- 通过“粒子跟随个体最优和群体最优”来逼近最优解
"""

import numpy as np

# ==========================================================
# 一、可调默认参数
# ==========================================================
SWARM_SIZE = 50
MAX_ITER = 150
W = 0.7              # 惯性权重
C1 = 1.5             # 个体学习因子
C2 = 1.5             # 社会学习因子
V_RATIO = 0.2        # 最大速度占变量范围比例
RANDOM_SEED = 42


def clip_x(x, bounds):
    """将粒子位置裁剪到上下界内"""
    x = x.copy()
    for i, (low, high) in enumerate(bounds):
        x[i] = np.clip(x[i], low, high)
    return x


def run_pso(
    objective_func,
    bounds,
    swarm_size=SWARM_SIZE,
    max_iter=MAX_ITER,
    w=W,
    c1=C1,
    c2=C2,
    v_ratio=V_RATIO,
    random_seed=RANDOM_SEED,
    record_indices=(0, 1)
):
    """
    运行粒子群算法
    """
    np.random.seed(random_seed)

    dim = len(bounds)

    # 初始化粒子位置
    positions = np.array([
        [np.random.uniform(low, high) for low, high in bounds]
        for _ in range(swarm_size)
    ], dtype=float)

    # 初始化粒子速度
    velocities = np.array([
        [np.random.uniform(-v_ratio * (high - low), v_ratio * (high - low)) for low, high in bounds]
        for _ in range(swarm_size)
    ], dtype=float)

    # 初始化个体最优
    pbest = positions.copy()
    pbest_result = [objective_func(x) for x in positions]
    pbest_fitness = np.array([r["fitness"] for r in pbest_result])

    # 初始化群体最优
    gbest_idx = np.argmin(pbest_fitness)
    gbest = pbest[gbest_idx].copy()
    gbest_result = pbest_result[gbest_idx].copy()

    history = []
    i1, i2 = record_indices

    for it in range(max_iter):
        for i in range(swarm_size):
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)

            # 更新速度
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (pbest[i] - positions[i])
                + c2 * r2 * (gbest - positions[i])
            )

            # 限制速度
            for j, (low, high) in enumerate(bounds):
                vmax = v_ratio * (high - low)
                velocities[i, j] = np.clip(velocities[i, j], -vmax, vmax)

            # 更新位置
            positions[i] = positions[i] + velocities[i]
            positions[i] = clip_x(positions[i], bounds)

            result = objective_func(positions[i])

            # 更新个体最优
            if result["fitness"] < pbest_fitness[i]:
                pbest[i] = positions[i].copy()
                pbest_result[i] = result.copy()
                pbest_fitness[i] = result["fitness"]

                # 更新群体最优
                if result["fitness"] < gbest_result["fitness"]:
                    gbest = positions[i].copy()
                    gbest_result = result.copy()

        history.append({
            "iter": it + 1,
            "p1": float(gbest[i1]),
            "p2": float(gbest[i2]),
            "fitness": float(gbest_result["fitness"]),
            "best_fitness": float(gbest_result["fitness"]),
            "cl": float(gbest_result["cl"]),
            "cd": float(gbest_result["cd"]),
            "t_max": float(gbest_result["t_max"]),
            "feasible": int(gbest_result["feasible"])
        })

    return {
        "algorithm": "PSO",
        "best_x": gbest,
        "best_result": gbest_result,
        "history": history
    }
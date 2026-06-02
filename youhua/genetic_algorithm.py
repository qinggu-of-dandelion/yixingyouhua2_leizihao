# -*- coding: utf-8 -*-
"""
genetic_algorithm.py
--------------------
本文件实现遗传算法（Genetic Algorithm, GA）。

适用场景：
- 连续变量优化
- 群体搜索
- 体现“选择-交叉-变异”的思想
"""

import numpy as np


# ==========================================================
# 一、可调默认参数
# ==========================================================
POP_SIZE = 120            # 种群规模
GENERATIONS = 150        # 迭代代数
CROSSOVER_RATE = 0.9     # 交叉概率
MUTATION_RATE = 0.2     # 变异概率
MUTATION_SCALE = 0.1    # 变异幅度，相对于变量范围
ELITE_SIZE = 3           # 精英保留数量
RANDOM_SEED = 42         # 随机种子


def clip_x(x, bounds):
    """将个体裁剪到上下界内"""
    x = x.copy()
    for i, (low, high) in enumerate(bounds):
        x[i] = np.clip(x[i], low, high)
    return x


def init_population(pop_size, bounds):
    """随机初始化种群"""
    pop = []
    for _ in range(pop_size):
        ind = [np.random.uniform(low, high) for low, high in bounds]
        pop.append(ind)
    return np.array(pop, dtype=float)


def crossover(p1, p2):
    """
    连续变量交叉
    这里采用线性混合方式
    """
    alpha = np.random.rand(len(p1))
    c1 = alpha * p1 + (1 - alpha) * p2
    c2 = alpha * p2 + (1 - alpha) * p1
    return c1, c2


def mutate(x, bounds, mutation_rate, mutation_scale):
    """高斯变异"""
    x = x.copy()
    for i, (low, high) in enumerate(bounds):
        if np.random.rand() < mutation_rate:
            x[i] += np.random.normal(0, mutation_scale * (high - low))
    return clip_x(x, bounds)


def select_one(population, fitnesses):
    """
    锦标赛选择
    随机挑 3 个个体，选 fitness 最小的
    """
    idx = np.random.choice(len(population), 3, replace=False)
    best_idx = idx[np.argmin(fitnesses[idx])]
    return population[best_idx].copy()


def run_ga(
    objective_func,
    bounds,
    pop_size=POP_SIZE,
    generations=GENERATIONS,
    crossover_rate=CROSSOVER_RATE,
    mutation_rate=MUTATION_RATE,
    mutation_scale=MUTATION_SCALE,
    elite_size=ELITE_SIZE,
    random_seed=RANDOM_SEED,
    record_indices=(0, 1)
):
    """
    运行遗传算法
    """
    np.random.seed(random_seed)

    # 初始化种群
    population = init_population(pop_size, bounds)
    fitnesses = np.array([objective_func(x)["fitness"] for x in population])

    # 初始化全局最优
    best_idx = np.argmin(fitnesses)
    best_x = population[best_idx].copy()
    best_result = objective_func(best_x)

    history = []
    i1, i2 = record_indices

    for gen in range(generations):
        # 精英保留
        elite_idx = np.argsort(fitnesses)[:elite_size]
        new_pop = [population[i].copy() for i in elite_idx]

        # 不断生成新个体直到种群满
        while len(new_pop) < pop_size:
            p1 = select_one(population, fitnesses)
            p2 = select_one(population, fitnesses)

            if np.random.rand() < crossover_rate:
                c1, c2 = crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()

            c1 = mutate(c1, bounds, mutation_rate, mutation_scale)
            c2 = mutate(c2, bounds, mutation_rate, mutation_scale)

            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        # 更新种群
        population = np.array(new_pop)
        fitnesses = np.array([objective_func(x)["fitness"] for x in population])

        # 记录这一代最优个体
        gen_best_idx = np.argmin(fitnesses)
        gen_best_x = population[gen_best_idx].copy()
        gen_best_result = objective_func(gen_best_x)

        # 更新全局最优
        if gen_best_result["fitness"] < best_result["fitness"]:
            best_x = gen_best_x.copy()
            best_result = gen_best_result.copy()

        history.append({
            "generation": gen + 1,
            "p1": float(gen_best_x[i1]),
            "p2": float(gen_best_x[i2]),
            "fitness": float(gen_best_result["fitness"]),
            "best_fitness": float(best_result["fitness"]),
            "cl": float(gen_best_result["cl"]),
            "cd": float(gen_best_result["cd"]),
            "t_max": float(gen_best_result["t_max"]),
            "feasible": int(gen_best_result["feasible"])
        })

    return {
        "algorithm": "GA",
        "best_x": best_x,
        "best_result": best_result,
        "history": history
    }
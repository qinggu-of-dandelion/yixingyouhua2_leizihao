"""
simulated_annealing.py
----------------------
本文件实现模拟退火算法（Simulated Annealing, SA）。

适用场景：
- 连续变量优化
- 本案例中用于 12 维 CST 参数优化
- 目标：通过代理模型预测结果，最小化 cd，同时满足 cl 和 t_max 约束

"""

import numpy as np

# ==========================================================
# 一、可调默认参数（如果主程序不传入）
# ==========================================================
MAX_ITER = 3000        # 最大迭代次数
INIT_TEMP = 1.0        # 初始温度
COOLING_RATE = 0.995   # 降温系数，越接近 1，降温越慢，搜索更充分
STEP_RATIO = 0.05      # 邻域步长比例，相对于变量范围
RANDOM_SEED = 42       # 随机种子，保证结果可重复

def clip_x(x, bounds):
    x = x.copy()
    for i, (low, high) in enumerate(bounds):
        x[i] = np.clip(x[i], low, high)
    return x


def get_neighbor(x, bounds, step_ratio):
    """
    在当前解附近随机生成一个邻域解
    思路：
    - 对每个变量加一个高斯随机扰动
    - 扰动大小与变量范围成比例
    """
    x_new = x.copy()
    for i, (low, high) in enumerate(bounds):
        step = np.random.normal(0, step_ratio * (high - low))
        x_new[i] += step
    return clip_x(x_new, bounds)


def run_sa(
    objective_func,
    bounds,
    max_iter=MAX_ITER,
    init_temp=INIT_TEMP,
    cooling_rate=COOLING_RATE,
    step_ratio=STEP_RATIO,
    random_seed=RANDOM_SEED,
    record_indices=(0, 1)
):
    """
    运行模拟退火算法
    objective_func : function
        目标函数，输入 x，输出一个 dict，至少包含：
        {
            "fitness": ...,
            "cl": ...,
            "cd": ...,
            "t_max": ...,
            "feasible": ...
        }
    bounds : list of tuple
        设计变量上下界
    max_iter : int
        最大迭代次数
    init_temp : float
        初始温度
    cooling_rate : float
        每一步温度乘以该系数进行降温
    step_ratio : float
        邻域扰动比例
    random_seed : int
        随机种子
    record_indices : tuple
        指定历史记录里保存哪两个参数，用于后续画探索路径图
        例如 (0, 1) 表示记录第 1 和第 2 个参数
    Returns
    -------
    dict
        包含最优解、最优结果、历史记录
    """
    np.random.seed(random_seed)

    # 1. 随机初始化一个起点
    x = np.array([np.random.uniform(low, high) for low, high in bounds], dtype=float)
    result = objective_func(x)

    # 2. 初始化全局最优
    best_x = x.copy()
    best_result = result.copy()

    temp = init_temp
    history = []

    i1, i2 = record_indices

    # 3. 开始迭代
    for i in range(max_iter):
        # 生成邻域点
        new_x = get_neighbor(x, bounds, step_ratio)
        new_result = objective_func(new_x)

        # fitness 越小越好
        delta = new_result["fitness"] - result["fitness"]

        # Metropolis 接受准则
        if delta < 0 or np.random.rand() < np.exp(-delta / max(temp, 1e-12)):
            x = new_x
            result = new_result

        # 更新全局最优
        if result["fitness"] < best_result["fitness"]:
            best_x = x.copy()
            best_result = result.copy()

        # 保存历史，用于画收敛图和路径图
        history.append({
            "iter": i + 1,
            "p1": float(x[i1]),
            "p2": float(x[i2]),
            "fitness": float(result["fitness"]),
            "best_fitness": float(best_result["fitness"]),
            "cl": float(result["cl"]),
            "cd": float(result["cd"]),
            "t_max": float(result["t_max"]),
            "feasible": int(result["feasible"])
        })

        # 降温
        temp *= cooling_rate

    return {
        "algorithm": "SA",
        "best_x": best_x,
        "best_result": best_result,
        "history": history
    }
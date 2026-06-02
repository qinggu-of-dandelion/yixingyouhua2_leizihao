# -*- coding: utf-8 -*-
"""
main_optimization.py
--------------------
主程序功能：
1. 读取代理模型（kriging / nn / svr）
2. 确定 CST 参数优化范围
3. 调用 SA / GA / PSO / RL 优化算法
4. 保存最优 CST 参数
5. 用 CST 重建优化后的翼型
6. 保存优化翼型 dat 文件
7. 绘制 baseline 与优化翼型对比图
8. 绘制收敛曲线、搜索路径图和 fitness 分布图
"""

import os
import json
from math import comb

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulated_annealing import run_sa
from genetic_algorithm import run_ga
from particle_swarm import run_pso
from reinforcement_learning import run_rl


# ==========================================================
# 一、可调参数区
# ==========================================================

# -----------------------------
# 1. 运行哪个优化算法
# 可选："SA" 模拟退火 / "GA" 遗传算法 / "PSO" 粒子群 / "RL" 强化学习 / "ALL" 全部对比
ALGORITHM = "RL"
# 2. 使用哪种代理模型
# 可选："kriging" / "nn" / "svr"
MODEL_TYPE = "kriging"
# 3. 路径设置
from pathlib import Path
# 当前 .py 文件所在目录：youhua
ROOT_DIR = Path(__file__).resolve().parent
# 项目总目录
PROJECT_DIR = ROOT_DIR.parent
# 代理模型文件夹
MODEL_DIR = PROJECT_DIR / "surrogate" / "surrogate_saved_models"
# 优化结果保存目录
SAVE_DIR = ROOT_DIR / "optimization_results"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
# 分目录保存结果
PLOT_DIR = os.path.join(SAVE_DIR, "plots")          # 保存所有图片
AIRFOIL_DIR = os.path.join(SAVE_DIR, "airfoils")    # 保存优化后的翼型 dat
DATA_DIR = os.path.join(SAVE_DIR, "data")           # 保存 csv / json / excel

# -----------------------------
# 4. baseline 的 CST 参数；如果后续更换基准翼型，改这里
# -----------------------------
AU_BASE = np.array([0.21828767, 0.23832163, 0.30765553, 0.19912590, 0.30210088, 0.25983377], dtype=float)
AL_BASE = np.array([-0.13451469, -0.06345951, -0.01589266, -0.07290444, 0.02177084, -0.04373223], dtype=float)
DZ_U = 0.0015574793
DZ_L = -0.0000386697
# CST 参数
N1 = 0.5
N2 = 1.0
N_POINTS = 301

# 5. 优化变量参数范围
DELTA_AU = 0.065     # 与 batch_run/config.py 中的采样扰动范围保持一致
DELTA_AL = 0.065     # 与 batch_run/config.py 中的采样扰动范围保持一致
SHRINK_RATIO = 0.05  # 优化范围相对采样范围的收缩比例

# 6. baseline 气动指标；更换工况或基准翼型时需要同步修改
BASELINE_CL = 0.719600
BASELINE_CD = 0.005640
BASELINE_TMAX = 0.120

# 7. 约束条件
# 目标：最小化 cd
# 约束：cl >= CL_MIN, t_max >= TMAX_MIN
CL_MIN = BASELINE_CL
TMAX_MIN = BASELINE_TMAX

# 8. 罚函数系数；如果最优结果仍不满足约束，可继续调大
PENALTY_CL = 1e7
PENALTY_TMAX = 1e7

# 9. 随机种子、绘图参数、保存开关与文件名前缀
RANDOM_SEED = 42

#   (0,1) -> Au_0, Au_1
#   (0,6) -> Au_0, Al_0
PLOT_PARAM_I = 0
PLOT_PARAM_J = 6
SAVE_PLOTS = True
CASE_NAME = "airfoil_opt"

# 10. SA 参数
SA_PARAMS = {
    "max_iter": 6000,
    "init_temp": 1.0,
    "cooling_rate": 0.997,
    "step_ratio": 0.04,  # 每次扰动步长占变量范围宽度的比例
    "random_seed": RANDOM_SEED,
    "record_indices": (PLOT_PARAM_I, PLOT_PARAM_J)
}

# 11. GA 参数
GA_PARAMS = {
    "pop_size": 150,
    "generations": 500,
    "crossover_rate": 0.9,
    "mutation_rate": 0.15,
    "mutation_scale": 0.07,
    "elite_size": 4,
    "random_seed": RANDOM_SEED,
    "record_indices": (PLOT_PARAM_I, PLOT_PARAM_J)
}

# 12. PSO 参数
PSO_PARAMS = {
    "swarm_size": 120,
    "max_iter": 500,
    "w": 0.72,
    "c1": 1.7,
    "c2": 1.7,
    "v_ratio": 0.10,
    "random_seed": RANDOM_SEED,
    "record_indices": (PLOT_PARAM_I, PLOT_PARAM_J)
}


# 二、基础设置
# ==========================================================

RL_PARAMS = {
    "episodes": 300,            # 训练总轮数（增加到 250 以匹配更多预算）
    "steps_per_episode": 35,    # 每轮最大步数
    "step_ratio": 0.06,         # 每步动作幅度占搜索空间宽度的比例（增大以增强探索）
    "gamma": 0.98,              # 折扣因子，控制未来奖励权重
    "tau": 0.01,                # 目标网络软更新系数
    "actor_lr": 1e-4,           # Actor 网络学习率
    "critic_lr": 3e-4,          # Critic 网络学习率
    "batch_size": 256,          # 经验回放采样批量大小
    "warmup_steps": 800,        # 随机探索预热步数
    "exploration_noise": 0.20,  # 初始探索噪声标准差（略微增大）
    "min_noise": 0.02,          # 探索噪声下限（保持最低探索能力）
    "noise_decay": 0.9985,      # 每轮噪声衰减系数（大幅减缓，避免过早丧失探索能力）
    "hidden_sizes": (128, 128), # 隐藏层神经元数量
    "random_seed": RANDOM_SEED, # 随机种子
    "record_indices": (PLOT_PARAM_I, PLOT_PARAM_J),  # 记录/绘图用的变量索引
    "updates_per_step": 2,      # 每步梯度更新次数（充分学习经验）
    "patience": 25,             # 连续多少轮无改善时触发探索重启
}

PARAM_NAMES = [f"Au_{i}" for i in range(6)] + [f"Al_{i}" for i in range(6)]
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
def load_obj(path):
    return joblib.load(path)
def build_bounds_from_base(au_base, al_base, delta_au, delta_al, shrink_ratio=0.0):
    """根据采样扰动范围生成优化边界。"""
    delta_au_opt = delta_au * (1.0 - shrink_ratio)
    delta_al_opt = delta_al * (1.0 - shrink_ratio)

    lower_bounds = np.concatenate([
        au_base - delta_au_opt,
        al_base - delta_al_opt
    ])
    upper_bounds = np.concatenate([
        au_base + delta_au_opt,
        al_base + delta_al_opt
    ])

    return [(float(lower_bounds[i]), float(upper_bounds[i])) for i in range(len(lower_bounds))]
BOUNDS = build_bounds_from_base(
    au_base=AU_BASE,
    al_base=AL_BASE,
    delta_au=DELTA_AU,
    delta_al=DELTA_AL,
    shrink_ratio=SHRINK_RATIO
)

def build_model_file_dict(model_dir, model_type):
    """拼接模型、输入 scaler、输出 scaler 和 meta 文件路径。"""
    return {
        "cl": {
            "model": os.path.join(model_dir, f"{model_type}_cl.pkl"),
            "x_scaler": os.path.join(model_dir, f"{model_type}_cl_x_scaler.pkl"),
            "y_scaler": os.path.join(model_dir, f"{model_type}_cl_y_scaler.pkl"),
            "meta": os.path.join(model_dir, f"{model_type}_cl_meta.json"),
        },
        "cd": {
            "model": os.path.join(model_dir, f"{model_type}_cd.pkl"),
            "x_scaler": os.path.join(model_dir, f"{model_type}_cd_x_scaler.pkl"),
            "y_scaler": os.path.join(model_dir, f"{model_type}_cd_y_scaler.pkl"),
            "meta": os.path.join(model_dir, f"{model_type}_cd_meta.json"),
        },
        "t_max": {
            "model": os.path.join(model_dir, f"{model_type}_t_max.pkl"),
            "x_scaler": os.path.join(model_dir, f"{model_type}_t_max_x_scaler.pkl"),
            "y_scaler": os.path.join(model_dir, f"{model_type}_t_max_y_scaler.pkl"),
            "meta": os.path.join(model_dir, f"{model_type}_t_max_meta.json"),
        }
    }

# 三、CST 相关函数
# ==========================================================

def class_function(x, n1=0.5, n2=1.0):
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 0.0, 1.0)
    return (x ** n1) * ((1 - x) ** n2)

def bernstein_poly(i, n, x):
    return comb(n, i) * (x ** i) * ((1 - x) ** (n - i))

def shape_function(x, A):
    n = len(A) - 1
    s = np.zeros_like(x, dtype=float)
    for i in range(n + 1):
        s += A[i] * bernstein_poly(i, n, x)
    return s

def cst_surface(x, A, n1=0.5, n2=1.0, dz=0.0):
    return class_function(x, n1, n2) * shape_function(x, A) + x * dz

def rebuild_airfoil(Au, Al, dz_u, dz_l, n_points=201, n1=0.5, n2=1.0):
    """根据 CST 参数重建上下表面。"""
    beta = np.linspace(0, np.pi, n_points)
    x = 0.5 * (1 - np.cos(beta))
    yu = cst_surface(x, Au, n1=n1, n2=n2, dz=dz_u)
    yl = cst_surface(x, Al, n1=n1, n2=n2, dz=dz_l)
    return x, yu, x, yl

def write_airfoil_dat(filename, name, xu, yu, xl, yl):
    """保存为 XFOIL 可读取的 dat 文件。"""
    upper = np.column_stack([xu, yu])[::-1]
    lower = np.column_stack([xl, yl])[1:]
    coords = np.vstack([upper, lower])

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"{name}\n")
        for x, y in coords:
            f.write(f"{x:.8f} {y:.8f}\n")

def calc_thickness(xu, yu, xl, yl, n_interp=500):
    """计算厚度分布和最大厚度。"""
    x_common = np.linspace(0.0, 1.0, n_interp)
    yu_i = np.interp(x_common, xu, yu)
    yl_i = np.interp(x_common, xl, yl)
    t = yu_i - yl_i
    idx = np.argmax(t)
    return {
        "x": x_common,
        "t": t,
        "t_max": float(t[idx]),
        "x_tmax": float(x_common[idx])
    }


# 四、代理模型评估器
# ==========================================================

class Evaluator:
    """分别加载 cl / cd / t_max 的模型、输入 scaler 和输出 scaler。"""

    def __init__(self, model_dir, model_type):
        self.file_dict = build_model_file_dict(model_dir, model_type)
        self.models = {}
        self.x_scalers = {}
        self.y_scalers = {}
        self.target_transforms = {}

        for target in ["cl", "cd", "t_max"]:
            self.models[target] = load_obj(self.file_dict[target]["model"])
            self.x_scalers[target] = load_obj(self.file_dict[target]["x_scaler"])
            self.y_scalers[target] = load_obj(self.file_dict[target]["y_scaler"])
            self.target_transforms[target] = "identity"

            meta_path = self.file_dict[target]["meta"]
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self.target_transforms[target] = meta.get("target_transform", "identity")

    def inverse_target_transform(self, y, target):
        transform = self.target_transforms.get(target, "identity")
        if transform == "identity":
            return y
        if transform == "log":
            return np.exp(y)
        raise ValueError(f"不支持的目标变换: {transform}")

    def predict_one_target(self, x, target):
        x = np.array(x, dtype=float).reshape(1, -1)
        x_scaled = self.x_scalers[target].transform(x)
        y_scaled = self.models[target].predict(x_scaled)
        y_scaled = np.array(y_scaled).reshape(-1, 1)
        y = self.y_scalers[target].inverse_transform(y_scaled)
        y = self.inverse_target_transform(y, target)
        return float(y.ravel()[0])

    def predict(self, x):
        return {
            "cl": self.predict_one_target(x, "cl"),
            "cd": self.predict_one_target(x, "cd"),
            "t_max": self.predict_one_target(x, "t_max")
        }

# 五、目标函数
# ==========================================================

def objective_function(x, evaluator):
    """
    目标：最小化 cd
    约束：cl >= CL_MIN, t_max >= TMAX_MIN
    """
    pred = evaluator.predict(x)
    cl = pred["cl"]
    cd = pred["cd"]
    t_max = pred["t_max"]

    penalty = 0.0
    if cl < CL_MIN:
        penalty += PENALTY_CL * (CL_MIN - cl) ** 2
    if t_max < TMAX_MIN:
        penalty += PENALTY_TMAX * (TMAX_MIN - t_max) ** 2

    fitness = cd + penalty

    return {
        "fitness": float(fitness),
        "cl": float(cl),
        "cd": float(cd),
        "t_max": float(t_max),
        "penalty": float(penalty),
        "feasible": bool(cl >= CL_MIN and t_max >= TMAX_MIN)
    }

# 六、绘图函数
# ==========================================================

def plot_convergence(history_df, algo_name):
    x_col = "iter" if "iter" in history_df.columns else "generation"
    y_col = "best_cd" if "best_cd" in history_df.columns else "cd"

    plt.figure(figsize=(8, 5))
    plt.plot(history_df[x_col], history_df[y_col], linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel(y_col)
    plt.title(f"{algo_name} Convergence")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{algo_name}_convergence.png"), dpi=300)
    plt.close()

def plot_search_path(history_df, algo_name):
    if "p1" not in history_df.columns or "p2" not in history_df.columns:
        return
    plt.figure(figsize=(7, 6))

    # 搜索路径连线
    plt.plot(history_df["p1"], history_df["p2"], linewidth=1.5)

    # 起点
    plt.scatter(
        history_df["p1"].iloc[0],
        history_df["p2"].iloc[0],
        s=60,
        marker="o",
        label="Start"
    )

    # 终点
    plt.scatter(
        history_df["p1"].iloc[-1],
        history_df["p2"].iloc[-1],
        s=80,
        marker="*",
        label="End"
    )

    plt.xlabel(PARAM_NAMES[PLOT_PARAM_I])
    plt.ylabel(PARAM_NAMES[PLOT_PARAM_J])
    plt.title(f"{algo_name} Search Path")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{algo_name}_search_path.png"), dpi=300)
    plt.close()


def plot_fitness_scatter(history_df, algo_name):
    if "p1" not in history_df.columns or "p2" not in history_df.columns:
        return

    plt.figure(figsize=(7, 6))
    sc = plt.scatter(history_df["p1"], history_df["p2"], c=history_df["cd"], s=35, alpha=0.8)
    plt.colorbar(sc, label="cd")
    plt.xlabel(PARAM_NAMES[PLOT_PARAM_I])
    plt.ylabel(PARAM_NAMES[PLOT_PARAM_J])
    plt.title(f"{algo_name} cd Scatter")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{algo_name}_cd_scatter.png"), dpi=300)
    plt.close()


def plot_all_convergence(results, file_tag):
    plt.figure(figsize=(8, 5))

    for result in results:
        history_df = pd.DataFrame(result["history"])
        x_col = "iter" if "iter" in history_df.columns else "generation"
        y_col = "best_cd" if "best_cd" in history_df.columns else "cd"
        plt.plot(history_df[x_col], history_df[y_col], linewidth=2, label=result["algorithm"])

    plt.xlabel("Iteration")
    plt.ylabel("best_cd / cd")
    plt.title("Algorithms Convergence Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{file_tag}_all_convergence.png"), dpi=300)
    plt.close()


def plot_baseline_vs_optimized(xu0, yu0, xl0, yl0, xu1, yu1, xl1, yl1, file_tag):
    plt.figure(figsize=(10, 4.5))
    plt.plot(xu0, yu0, color="red", lw=2.5, label="Baseline")
    plt.plot(xl0, yl0, color="red", lw=2.5)
    plt.plot(xu1, yu1, color="blue", lw=2.0, label="Optimized")
    plt.plot(xl1, yl1, color="blue", lw=2.0)
    plt.axis("equal")
    plt.xlabel("x/c")
    plt.ylabel("y/c")
    plt.title("Baseline vs Optimized Airfoil")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{file_tag}_baseline_vs_optimized.png"), dpi=300)
    plt.close()

# ==========================================================
# 七、结果保存
# ==========================================================

def save_best_airfoil(best_x, file_tag):
    """
    将最优的 12 个 CST 参数拆成 Au / Al，
    然后重建翼型并保存 dat 文件与对比图。
    """
    Au_opt = np.array(best_x[:6], dtype=float)
    Al_opt = np.array(best_x[6:], dtype=float)

    xu0, yu0, xl0, yl0 = rebuild_airfoil(AU_BASE, AL_BASE, DZ_U, DZ_L, n_points=N_POINTS, n1=N1, n2=N2)
    xu1, yu1, xl1, yl1 = rebuild_airfoil(Au_opt, Al_opt, DZ_U, DZ_L, n_points=N_POINTS, n1=N1, n2=N2)

    th_base = calc_thickness(xu0, yu0, xl0, yl0)
    th_opt = calc_thickness(xu1, yu1, xl1, yl1)

    # dat 文件放到 airfoils 文件夹
    dat_path = os.path.join(AIRFOIL_DIR, f"{file_tag}_optimized_airfoil.dat")
    write_airfoil_dat(dat_path, f"{file_tag}_optimized", xu1, yu1, xl1, yl1)

    if SAVE_PLOTS:
        plot_baseline_vs_optimized(xu0, yu0, xl0, yl0, xu1, yu1, xl1, yl1, file_tag)

    geom_data = {
        "Au_opt": Au_opt.tolist(),
        "Al_opt": Al_opt.tolist(),
        "baseline_t_max_geom": th_base["t_max"],
        "optimized_t_max_geom": th_opt["t_max"],
        "optimized_airfoil_dat": dat_path
    }

    # 几何相关 json 放到 data 文件夹
    with open(os.path.join(DATA_DIR, f"{file_tag}_optimized_cst_geometry.json"), "w", encoding="utf-8") as f:
        json.dump(geom_data, f, ensure_ascii=False, indent=4)

    return geom_data


def save_result(result):
    algo = result["algorithm"]
    best_x = result["best_x"]
    best_result = result["best_result"]
    history = result["history"]

    file_tag = f"{CASE_NAME}_{MODEL_TYPE}_{algo}"

    # 只保存 baseline 和优化结果的 cl / cd / t_max
    row = {
        "algorithm": algo,
        "baseline_cl": BASELINE_CL,
        "baseline_cd": BASELINE_CD,
        "baseline_t_max": BASELINE_TMAX,
        "opt_cl": best_result["cl"],
        "opt_cd": best_result["cd"],
        "opt_t_max": best_result["t_max"]
    }

    # 单个算法结果表
    pd.DataFrame([row]).to_csv(
        os.path.join(DATA_DIR, f"{file_tag}_best_result.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    with open(os.path.join(DATA_DIR, f"{file_tag}_best_result.json"), "w", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False, indent=4)

    # 保留 history，供后续绘图和过程分析使用
    history_df = pd.DataFrame(history)
    history_df.to_csv(
        os.path.join(DATA_DIR, f"{file_tag}_history.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # 保存几何结果
    geom_data = save_best_airfoil(best_x, file_tag)

    # 保存优化过程图
    if SAVE_PLOTS:
        plot_convergence(history_df, file_tag)
        plot_search_path(history_df, file_tag)
        plot_fitness_scatter(history_df, file_tag)

    return {
        "file_tag": file_tag,
        "row": row,
        "geom_data": geom_data
    }

def print_result(result):
    print("\n" + "=" * 60)
    print("模型类型：", MODEL_TYPE)
    print("算法：", result["algorithm"])
    print("最优参数：")
    print(result["best_x"])
    print("预测 cl =", result["best_result"]["cl"])
    print("预测 cd =", result["best_result"]["cd"])
    print("预测 t_max =", result["best_result"]["t_max"])
    print("fitness =", result["best_result"]["fitness"])
    print("penalty =", result["best_result"]["penalty"])
    print("是否满足约束 =", result["best_result"]["feasible"])
    print("=" * 60)


# ==========================================================
# 八、主程序
# ==========================================================

def main(baseline_cl=None, baseline_cd=None, baseline_t_max=None,
         au_base=None, al_base=None, dz_u=None, dz_l=None,
         seed_name=None,
         algorithm=None, model_type=None,
         save_dir=None):
    """
    优化主函数。

    参数（全部可选，默认使用模块顶部的硬编码值）:
        baseline_cl/cd/t_max: 基线气动性能（用于约束条件）
        au_base, al_base: 基线 CST 系数
        dz_u, dz_l: 尾缘项
        seed_name: 种子名称（用于输出文件命名和对比图）
        algorithm: 优化算法 ("SA"/"GA"/"PSO"/"RL"/"ALL")
        model_type: 代理模型 ("kriging"/"nn"/"svr")
        save_dir: 自定义输出根目录
    """
    # 使用传入值或默认值
    _baseline_cl = BASELINE_CL if baseline_cl is None else float(baseline_cl)
    _baseline_cd = BASELINE_CD if baseline_cd is None else float(baseline_cd)
    _baseline_tmax = BASELINE_TMAX if baseline_t_max is None else float(baseline_t_max)
    _au_base = AU_BASE if au_base is None else np.asarray(au_base, dtype=float)
    _al_base = AL_BASE if al_base is None else np.asarray(al_base, dtype=float)
    _dz_u = DZ_U if dz_u is None else float(dz_u)
    _dz_l = DZ_L if dz_l is None else float(dz_l)
    _algo = ALGORITHM if algorithm is None else algorithm
    _model_type = MODEL_TYPE if model_type is None else model_type

    # 更新全局变量（用于 objective_function 和 save_best_airfoil 等函数引用）
    global CL_MIN, TMAX_MIN, BASELINE_CL, BASELINE_CD, BASELINE_TMAX
    global AU_BASE, AL_BASE, DZ_U, DZ_L
    global CASE_NAME
    CL_MIN = _baseline_cl
    TMAX_MIN = _baseline_tmax
    BASELINE_CL = _baseline_cl
    BASELINE_CD = _baseline_cd
    BASELINE_TMAX = _baseline_tmax
    AU_BASE = _au_base
    AL_BASE = _al_base
    DZ_U = _dz_u
    DZ_L = _dz_l

    if seed_name is not None:
        CASE_NAME = f"airfoil_opt_{seed_name}"
    else:
        CASE_NAME = "airfoil_opt"

    # 自定义输出目录
    if save_dir is not None:
        _save_dir = Path(save_dir)
        _plot_dir = _save_dir / "plots"
        _airfoil_dir = _save_dir / "airfoils"
        _data_dir = _save_dir / "data"
    else:
        _save_dir = SAVE_DIR
        _plot_dir = Path(PLOT_DIR)
        _airfoil_dir = Path(AIRFOIL_DIR)
        _data_dir = Path(DATA_DIR)

    ensure_dir(_save_dir)
    ensure_dir(_plot_dir)
    ensure_dir(_airfoil_dir)
    ensure_dir(_data_dir)

    # 重新计算 BOUNDS（因为 baseline CST 可能已变更）
    _bounds = build_bounds_from_base(
        au_base=_au_base,
        al_base=_al_base,
        delta_au=DELTA_AU,
        delta_al=DELTA_AL,
        shrink_ratio=SHRINK_RATIO
    )

    print(f"MODEL_TYPE = {_model_type}")
    print(f"ALGORITHM  = {_algo}")
    print(f"SAVE_DIR   = {_save_dir}")
    print(f"BASELINE   = CL:{_baseline_cl:.4f}, CD:{_baseline_cd:.6f}, TMAX:{_baseline_tmax:.4f}")

    evaluator = Evaluator(MODEL_DIR, _model_type)
    obj_func = lambda x: objective_function(x, evaluator)

    results = []

    if _algo == "SA":
        results.append(run_sa(obj_func, _bounds, **SA_PARAMS))
    elif _algo == "GA":
        results.append(run_ga(obj_func, _bounds, **GA_PARAMS))
    elif _algo == "PSO":
        results.append(run_pso(obj_func, _bounds, **PSO_PARAMS))
    elif _algo == "RL":
        results.append(run_rl(obj_func, _bounds, **RL_PARAMS))
    elif _algo == "ALL":
        results.append(run_sa(obj_func, _bounds, **SA_PARAMS))
        results.append(run_ga(obj_func, _bounds, **GA_PARAMS))
        results.append(run_pso(obj_func, _bounds, **PSO_PARAMS))
        results.append(run_rl(obj_func, _bounds, **RL_PARAMS))
    else:
        raise ValueError("ALGORITHM must be 'SA' / 'GA' / 'PSO' / 'RL' / 'ALL'")

    summary_rows = []

    for result in results:
        print_result(result)
        # 临时覆盖保存路径
        global PLOT_DIR, AIRFOIL_DIR, DATA_DIR
        _orig_plot, _orig_airfoil, _orig_data = PLOT_DIR, AIRFOIL_DIR, DATA_DIR
        PLOT_DIR, AIRFOIL_DIR, DATA_DIR = str(_plot_dir), str(_airfoil_dir), str(_data_dir)
        saved = save_result(result)
        PLOT_DIR, AIRFOIL_DIR, DATA_DIR = _orig_plot, _orig_airfoil, _orig_data

        summary_rows.append({
            "algorithm": result["algorithm"],
            "baseline_cl": BASELINE_CL,
            "baseline_cd": BASELINE_CD,
            "baseline_t_max": BASELINE_TMAX,
            "opt_cl": result["best_result"]["cl"],
            "opt_cd": result["best_result"]["cd"],
            "opt_t_max": result["best_result"]["t_max"]
        })

    summary_df = pd.DataFrame(summary_rows)

    # 保存 summary.csv
    summary_csv_path = _data_dir / f"{CASE_NAME}_{_model_type}_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")

    # 保存 summary.xlsx
    summary_excel_path = _data_dir / f"{CASE_NAME}_{_model_type}_summary.xlsx"
    try:
        with pd.ExcelWriter(summary_excel_path, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="summary", index=False)
            for result in results:
                algo = result["algorithm"]
                detail_row = {
                    "algorithm": algo,
                    "baseline_cl": BASELINE_CL,
                    "baseline_cd": BASELINE_CD,
                    "baseline_t_max": BASELINE_TMAX,
                    "opt_cl": result["best_result"]["cl"],
                    "opt_cd": result["best_result"]["cd"],
                    "opt_t_max": result["best_result"]["t_max"]
                }
                pd.DataFrame([detail_row]).to_excel(writer, sheet_name=algo, index=False)
    except Exception:
        pass  # Excel 写入失败时不影响整体流程

    if SAVE_PLOTS and len(results) > 1:
        plot_all_convergence(results, f"{CASE_NAME}_{_model_type}")

    print("\n优化完成，结果已保存到：")
    print(_save_dir)

    return results, summary_df


if __name__ == "__main__":
    main()

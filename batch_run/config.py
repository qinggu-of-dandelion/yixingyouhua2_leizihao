from pathlib import Path
import numpy as np

# =========================================================
# 路径设置
# =========================================================
# 当前 Python 文件所在文件夹
ROOT_DIR = Path(__file__).resolve().parent

AIRFOIL_DIR = ROOT_DIR / "airfoils"
PLOT_DIR = ROOT_DIR / "plots"
POLAR_DIR = ROOT_DIR / "polars"
XFOIL_EXE = AIRFOIL_DIR / "xfoil.exe"

# =========================================================
# baseline: NACA4412 的 CST 参数
# =========================================================
AU_BASE = np.array([0.21828767, 0.23832163, 0.30765553, 0.19912590, 0.30210088, 0.25983377])
AL_BASE = np.array([-0.13451469, -0.06345951, -0.01589266, -0.07290444, 0.02177084, -0.04373223])

DZ_U = 0.0015574793
DZ_L = -0.0000386697

# =========================================================
# CST / 几何参数
# =========================================================
N1 = 0.5
N2 = 1.0
N_POINTS = 201

# =========================================================
# LHS 采样参数
# =========================================================
# 下一轮建议使用 1500 个样本：
# - 比原来的 1000 个样本更适合观察“样本数增加后代理模型是否变准”
# - XFOIL 计算量仍然可控
N_SAMPLES = 1500
RANDOM_SEED = 42

# 采样参数扰动范围（绝对扰动）。
# 建议先用 0.065 做局部优化代理模型：
# - 0.075 覆盖更宽，但代理模型更难学准
# - 0.065 更聚焦在 NACA4412 附近，通常更适合当前“减阻且保持升力/厚度”的局部优化
# 如果后续想做更大范围全局探索，再恢复到 0.075 或更大。
DELTA_AU = 0.065
DELTA_AL = 0.065

# 是否固定 dz
FIX_DZ = True

# =========================================================
# 几何筛选参数
# =========================================================
T_MIN_THRESHOLD = 1e-5
X_MARGIN_CHECK = 0.02
ENFORCE_TMAX_NOT_SMALLER = False

# =========================================================
# XFOIL 求解参数
# =========================================================
RE = 6e6
MACH = 0.3
ITER = 200
ALPHA_LIST = [2.0]   #####注意更改
TARGET_ALPHA = 2.0   #####注意更改

# =========================================================
# 绘图参数
# =========================================================
N_SHOW_PLOTS = 12

# 是否自动打开保存后的图片
AUTO_OPEN_PLOTS = True


# =========================================================
# 从种子 JSON 加载基线参数（KMeans 多种子集成用）
# =========================================================
def load_baseline_from_seed_json(json_path):
    """
    从 fit_cst_for_seed.py 输出的 JSON 文件加载基线 CST 参数。

    用法:
        from config import load_baseline_from_seed_json
        seed = load_baseline_from_seed_json("seed_info.json")
        # seed['au'] -> np.array, seed['al'] -> np.array ...
    """
    import json
    import numpy as np

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        "au": np.array(data["au"], dtype=float),
        "al": np.array(data["al"], dtype=float),
        "dz_u": float(data["dz_u"]),
        "dz_l": float(data["dz_l"]),
        "t_max": float(data.get("t_max", np.nan)),
        "cl": float(data.get("cl", np.nan)) if data.get("cl") is not None else None,
        "cd": float(data.get("cd", np.nan)) if data.get("cd") is not None else None,
        "name": data.get("name", ""),
        "dat_path": data.get("dat_path", ""),
    }

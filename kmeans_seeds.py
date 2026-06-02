# -*- coding: utf-8 -*-
"""
kmeans_seeds.py — 基于 KMeans 聚类的翼型种子筛选模块（独立抽取版）
=====================================================================

功能概述：
  从一批翼型数据中，自动筛选出"可行且优秀"的翼型，用 KMeans 聚类
  识别多种不同的几何模式，每类输出一个代表性翼型作为"种子"。

  典型用途：作为翼型优化的前置步骤，为下游优化器提供多样化的初始翼型。

核心流程：
  1. 读取 CSV（含 CST 系数 + Cl/Cd 气动数据）
  2. 从 .dat 文件提取几何特征（最大厚度 t_max、最大弯度 camber_max）
  3. 可行性筛选（厚度、Cl、Cd 门槛）
  4. 质量标注（L/D 前 N% 分位数 → "优秀"）
  5. 对"优秀"翼型的 CST+几何特征做 StandardScaler + KMeans 聚类
  6. 每簇选离中心最近的样本作为种子
  7. 输出种子 .dat 文件 + 汇总 CSV

使用方式：
  -------- 方式1: Python API --------
  from kmeans_seeds import KMeansSeedSelector

  selector = KMeansSeedSelector(
      csv_path="airfoils.csv",
      dat_dir="./airfoil_dat_files/",
      output_dir="./seeds_output/",
      aoa=4.0,            # 攻角筛选
      n_clusters=3,       # 聚类簇数
  )
  seeds_df, summary_df = selector.run()

  -------- 方式2: 命令行 --------
  python kmeans_seeds.py --csv airfoils.csv --dat_dir ./dat_files/ -o ./seeds/ -k 3

依赖：numpy, pandas, scikit-learn, pyyaml (可选，仅CLI用)
"""

from __future__ import annotations

import math
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ============================================================================
# Section 1: .dat 文件解析 & 几何特征提取
# ============================================================================

_NUM_LINE = re.compile(r"^\s*[-+0-9\.eE]+\s+[-+0-9\.eE]+\s*$")


@dataclass
class AirfoilCoords:
    """翼型坐标数据结构"""
    x: np.ndarray   # 弦向坐标 (0~1)
    y: np.ndarray   # 法向坐标
    path: Path      # 来源文件路径


def _try_parse_xy(line: str) -> Optional[Tuple[float, float]]:
    """尝试解析一行 x y 坐标"""
    line = line.strip()
    if not line:
        return None
    if not _NUM_LINE.match(line):
        return None
    parts = line.split()
    if len(parts) < 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except Exception:
        return None


def load_dat_coords(dat_path: Path) -> AirfoilCoords:
    """从 .dat 文件加载翼型坐标，自动归一化弦长到 [0,1]"""
    xs: List[float] = []
    ys: List[float] = []
    with dat_path.open("r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            xy = _try_parse_xy(ln)
            if xy is None:
                continue
            x, y = xy
            xs.append(x)
            ys.append(y)

    if len(xs) < 20:
        raise ValueError(f"坐标点太少 ({len(xs)}): {dat_path}")

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)

    # 归一化弦长到 [0, 1]
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    chord = x_max - x_min
    if chord <= 0:
        raise ValueError(f"无效弦长: {dat_path}")
    x = (x - x_min) / chord

    return AirfoilCoords(x=x, y=y, path=dat_path)


def split_upper_lower(
    coords: AirfoilCoords
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    分割上下表面。处理 LE→TE→LE 和 TE→LE→TE 两种常见点序。
    返回: (x_upper, y_upper, x_lower, y_lower)
    """
    x, y = coords.x, coords.y
    n = len(x)

    i_le = int(np.argmin(x))
    i_te = int(np.argmax(x))

    is_le_first_or_last = (i_le == 0 or i_le == n - 1)
    is_te_in_middle = (0 < i_te < n - 1)

    if is_le_first_or_last and is_te_in_middle:
        # LE→TE→LE 点序：在 TE 处分割
        x1, y1 = x[: i_te + 1].copy(), y[: i_te + 1].copy()
        x2, y2 = x[i_te:].copy(), y[i_te:].copy()
        x2, y2 = x2[::-1], y2[::-1]  # 反向使 x 从小到大
        xu, yu = x1, y1
        xl, yl = x2, y2
    else:
        # TE→LE→TE 点序：在 LE 处分割
        xu = x[: i_le + 1].copy()
        yu = y[: i_le + 1].copy()
        xl = x[i_le:].copy()
        yl = y[i_le:].copy()

        if len(xu) > 1 and xu[0] > xu[-1]:
            xu, yu = xu[::-1], yu[::-1]
        if len(xl) > 1 and xl[0] > xl[-1]:
            xl, yl = xl[::-1], yl[::-1]

    # 判断上下表面：x≈0.5 处 y 值较大者为上表面
    try:
        iu = int(np.argmin(np.abs(xu - 0.5)))
        il = int(np.argmin(np.abs(xl - 0.5)))
        if yu[iu] < yl[il]:
            xu, xl = xl, xu
            yu, yl = yl, yu
    except Exception:
        pass

    return xu, yu, xl, yl


def max_thickness(coords: AirfoilCoords, n_grid: int = 400) -> float:
    """计算最大厚度比 t_max = max(y_upper(x) - y_lower(x))"""
    xu, yu, xl, yl = split_upper_lower(coords)

    x0 = max(float(np.min(xu)), float(np.min(xl)))
    x1 = min(float(np.max(xu)), float(np.max(xl)))
    if x1 <= x0:
        raise ValueError(f"上下表面 x 范围无重叠: {coords.path}")

    grid = np.linspace(x0, x1, n_grid)
    yu_i = np.interp(grid, xu, yu)
    yl_i = np.interp(grid, xl, yl)

    return float(np.max(yu_i - yl_i))


def max_camber(coords: AirfoilCoords, n_grid: int = 400) -> float:
    """计算最大弯度 camber_max = max(|(y_upper + y_lower)/2|)"""
    xu, yu, xl, yl = split_upper_lower(coords)

    x0 = max(float(np.min(xu)), float(np.min(xl)))
    x1 = min(float(np.max(xu)), float(np.max(xl)))
    if x1 <= x0:
        return np.nan

    grid = np.linspace(x0, x1, n_grid)
    yu_i = np.interp(grid, xu, yu)
    yl_i = np.interp(grid, xl, yl)

    camber_line = (yu_i + yl_i) / 2.0
    return float(np.max(np.abs(camber_line)))


def find_dat_file(filename: str, dat_root: Path) -> Optional[Path]:
    """
    根据 CSV 中的 Filename 查找对应的 .dat 文件。
    支持多种命名变体（带/不带 .dat 后缀、大小写等）。
    """
    dat_root = Path(dat_root)
    base = Path(str(filename)).name

    candidates = []
    if base.lower().endswith(".dat"):
        candidates.append(base)
    else:
        candidates.append(base + ".dat")
        candidates.append(base + ".DAT")
        candidates.append(base)

    # 直接查找
    for c in candidates:
        p = dat_root / c
        if p.exists():
            return p

    # 递归查找
    for c in candidates:
        matches = list(dat_root.rglob(c))
        if matches:
            matches.sort(key=lambda x: len(str(x)))
            return matches[0]

    # 按 stem 模糊匹配
    stem = Path(base).stem.lower()
    if stem:
        matches = [p for p in dat_root.rglob("*.dat") if p.stem.lower() == stem]
        if matches:
            matches.sort(key=lambda x: len(str(x)))
            return matches[0]

    return None


# ============================================================================
# Section 2: CST 参数化工具（用于从 CST 系数重构 .dat 文件）
# ============================================================================

def _cst_class_function(x: np.ndarray, n1: float = 0.5, n2: float = 1.0) -> np.ndarray:
    """CST 类函数: C(x) = x^N1 * (1-x)^N2"""
    return np.power(x, n1) * np.power(1.0 - x, n2)


def _cst_shape_function(x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """CST 形函数: 伯恩斯坦多项式加权和"""
    n = len(coeffs) - 1
    if n < 0:
        raise ValueError("CST 系数为空")
    s = np.zeros_like(x, dtype=float)
    for i, a in enumerate(coeffs):
        b = math.comb(n, i)
        s += a * b * np.power(x, i) * np.power(1.0 - x, n - i)
    return s


def cst_surface_y(x: np.ndarray, coeffs: np.ndarray,
                  n1: float = 0.5, n2: float = 1.0) -> np.ndarray:
    """CST 表面 y 坐标: y(x) = C(x) * S(x)"""
    return _cst_class_function(x, n1=n1, n2=n2) * _cst_shape_function(x, coeffs)


def cst_to_airfoil_coords(cst_coeffs: np.ndarray,
                          n_points: int = 201) -> AirfoilCoords:
    """
    从 8 维 CST 系数重构翼型坐标。
    前 4 个系数 = 上表面，后 4 个 = 下表面。
    输出点序: TE(1,0) → 上表面 → LE(0,0) → 下表面 → TE(1,0)
    """
    coeffs = np.asarray(cst_coeffs, dtype=float).flatten()
    if len(coeffs) != 8:
        raise ValueError(f"CST 系数长度必须为 8（上4+下4），当前为 {len(coeffs)}")

    cu = coeffs[:4]  # 上表面 CST
    cl = coeffs[4:]  # 下表面 CST

    # 余弦分布采样
    theta = np.linspace(0.0, np.pi, n_points)
    x_cos = 0.5 * (1.0 - np.cos(theta))

    yu = cst_surface_y(x_cos, cu)
    yl = cst_surface_y(x_cos, cl)

    # TE → 上表面 → LE → 下表面 → TE (Selig 格式)
    x_all = np.concatenate([x_cos[::-1], x_cos[1:]])
    y_all = np.concatenate([yu[::-1], yl[1:]])

    return AirfoilCoords(x=x_all, y=y_all, path=Path("CST"))


def write_dat_file(coords: AirfoilCoords, output_path: Path,
                   name: str = "airfoil") -> None:
    """将翼型坐标写入 .dat 文件（Selig 格式）"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"{name}\n")
        for x, y in zip(coords.x, coords.y):
            f.write(f"  {x:.10f}  {y:.10f}\n")


# ============================================================================
# Section 3: 可行性筛选 & 质量标注
# ============================================================================

def apply_feasibility_filter(
    df: pd.DataFrame,
    min_thickness: float = 0.08,
    min_cl: float = 0.3,
    min_cd: float = 0.001,
) -> pd.DataFrame:
    """
    可行性筛选:
      - t_max >= min_thickness（厚度足够）
      - Cl >= min_cl（有足够升力）
      - Cd > min_cd（阻力不为零/负）
      - t_max 不为 NaN
    """
    df2 = df.copy()
    n_before = len(df2)

    if "t_max" in df2.columns:
        df2 = df2[df2["t_max"].notna()]
        df2 = df2[df2["t_max"] >= min_thickness]

    if "Cl" in df2.columns:
        df2 = df2[df2["Cl"] >= min_cl]

    if "Cd" in df2.columns:
        df2 = df2[df2["Cd"] > min_cd]

    return df2


def classify_quality(
    df_feasible: pd.DataFrame,
    excellent_percentile: float = 0.70,
    ld_column: str = "L_over_D",
) -> Tuple[pd.DataFrame, Dict]:
    """
    基于 L/D 分位数做二分类:
      - L/D >= excellent_percentile 分位数 → "优秀"
      - 其余 → "合格"

    返回: (带 Quality 列的 DataFrame, 元数据字典)
    """
    if ld_column not in df_feasible.columns:
        # 尝试计算
        if "Cl" in df_feasible.columns and "Cd" in df_feasible.columns:
            df_feasible = df_feasible.copy()
            df_feasible[ld_column] = df_feasible["Cl"] / df_feasible["Cd"]
        else:
            raise ValueError(f"缺少 {ld_column} 列，且无法从 Cl/Cd 计算")

    threshold = float(df_feasible[ld_column].quantile(excellent_percentile))

    df_labeled = df_feasible.copy()
    df_labeled["Quality"] = "合格"
    df_labeled.loc[df_labeled[ld_column] >= threshold, "Quality"] = "优秀"

    meta = {
        "quality_rule": "binary",
        "excellent_definition": f"L/D >= {excellent_percentile*100:.0f}th percentile of feasible samples",
        "threshold_excellent_L_over_D": threshold,
        "n_feasible": int(len(df_feasible)),
        "n_excellent": int((df_labeled["Quality"] == "优秀").sum()),
        "n_qualified": int((df_labeled["Quality"] == "合格").sum()),
    }

    return df_labeled, meta


# ============================================================================
# Section 4: KMeans 聚类 & 种子选择
# ============================================================================

def _resolve_cst_cols(df: pd.DataFrame) -> List[str]:
    """自动识别 CST 系数列名"""
    # 格式1: "CST Coeff 1" ~ "CST Coeff 8"
    cols_a = [f"CST Coeff {i}" for i in range(1, 9)]
    if all(c in df.columns for c in cols_a):
        return cols_a

    # 格式2: "CST_Coeff_1" ~ "CST_Coeff_8"
    cols_b = [f"CST_Coeff_{i}" for i in range(1, 9)]
    if all(c in df.columns for c in cols_b):
        return cols_b

    # 格式3: 任意以 "cst" 开头的列
    guess = [c for c in df.columns if c.lower().startswith("cst")]
    if len(guess) >= 8:
        return guess[:8]

    raise ValueError(
        "找不到完整的 CST 系数列（期望 8 维）。\n"
        f"当前列名: {list(df.columns)}\n"
        "支持的命名格式: 'CST Coeff 1'~'CST Coeff 8' 或 'CST_Coeff_1'~'CST_Coeff_8'"
    )


def prepare_clustering_features(
    df: pd.DataFrame,
    use_cst: bool = True,
    use_thickness: bool = True,
    use_camber: bool = True,
    use_aerodynamic: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    """
    准备聚类特征矩阵（含 StandardScaler 标准化）。

    参数:
        df: 待聚类的 DataFrame
        use_cst: 是否使用 CST 系数 1~8
        use_thickness: 是否使用 t_max
        use_camber: 是否使用 camber_max
        use_aerodynamic: 是否使用 Cl, Cd, L_over_D

    返回:
        X: 标准化后的特征矩阵 (n_samples, n_features)
        feat_cols: 使用的特征列名列表
    """
    from sklearn.preprocessing import StandardScaler

    cols: List[str] = []

    if use_cst:
        cols.extend(_resolve_cst_cols(df))

    if use_thickness and "t_max" in df.columns:
        cols.append("t_max")

    if use_camber and "camber_max" in df.columns:
        cols.append("camber_max")

    if use_aerodynamic:
        for c in ["Cl", "Cd", "L_over_D"]:
            if c in df.columns:
                cols.append(c)

    if not cols:
        raise ValueError("聚类特征为空！请至少启用一种特征类型。")

    X = df[cols].values.astype(float)
    X = StandardScaler().fit_transform(X)
    return X, cols


def print_k_diagnostics(
    df: pd.DataFrame,
    X: np.ndarray,
    k_min: int = 2,
    k_max: int = 5,
    random_state: int = 42,
) -> None:
    """打印 k=2~5 的簇内 L/D 均值与样本数，辅助选择 k 值"""
    from sklearn.cluster import KMeans

    if "L_over_D" not in df.columns:
        print("[WARN] 无 L_over_D 列，跳过 k 诊断。")
        return

    print("\n" + "=" * 60)
    print("KMeans k 值诊断 (k=2~5 簇内 L/D 均值与样本数)")
    print("=" * 60)
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        for cid in range(k):
            mask = labels == cid
            n = int(mask.sum())
            ld_mean = float(df.loc[mask, "L_over_D"].mean()) if n > 0 else float("nan")
            print(f"  k={k} | cluster {cid + 1}: n={n:4d}, L/D_mean={ld_mean:.4f}")


def cluster_and_select_seeds(
    df_labeled: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = 42,
    use_cst: bool = True,
    use_thickness: bool = True,
    use_camber: bool = True,
    use_aerodynamic: bool = False,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    核心聚类 + 种子选择函数。

    参数:
        df_labeled: 包含 Quality 列的标注数据
        n_clusters: KMeans 聚类数
        random_state: 随机种子
        use_cst/thickness/camber/aerodynamic: 聚类特征开关
        verbose: 是否打印聚类信息

    返回:
        (good_clusters_df, summary_df, seeds_df)
        - good_clusters_df: 优秀样本 + cluster_id 列
        - summary_df: 每簇统计信息（样本数、L/D均值、CST均值等）
        - seeds_df: 每簇种子样本（最接近簇中心的真实翼型）
    """
    from sklearn.cluster import KMeans

    if "Quality" not in df_labeled.columns:
        raise ValueError("输入 DataFrame 缺少 'Quality' 列，请先运行 classify_quality()。")

    df_best = df_labeled[df_labeled["Quality"] == "优秀"].copy()
    if len(df_best) == 0:
        raise ValueError('没有「优秀」样本，无法聚类。请放宽可行性阈值或降低优秀分位数。')

    if verbose:
        print(f"\n聚类样本数: {len(df_best)} (优秀翼型)")

    # 准备特征
    X, feat_cols = prepare_clustering_features(
        df_best,
        use_cst=use_cst,
        use_thickness=use_thickness,
        use_camber=use_camber,
        use_aerodynamic=use_aerodynamic,
    )

    if verbose:
        print(f"聚类特征 ({len(feat_cols)} 维): {feat_cols}")

    # k 值诊断
    if verbose:
        print_k_diagnostics(df_best, X, random_state=random_state)

    # KMeans 聚类
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)
    df_best["cluster_id"] = labels + 1  # 簇编号从 1 开始

    if verbose:
        print(f"\nKMeans (k={n_clusters}) 聚类结果:")
        for cid in sorted(df_best["cluster_id"].unique()):
            n = int((df_best["cluster_id"] == cid).sum())
            ld = float(df_best[df_best["cluster_id"] == cid]["L_over_D"].mean())
            print(f"  Cluster {cid}: {n} 个样本, L/D 均值 = {ld:.4f}")

    # ----- 生成每簇汇总 -----
    cst_cols = _resolve_cst_cols(df_best)
    summary_rows = []
    for cid in sorted(df_best["cluster_id"].unique()):
        sub = df_best[df_best["cluster_id"] == cid]
        row = {
            "cluster_id": int(cid),
            "n_samples": int(len(sub)),
            "L_over_D_mean": float(sub["L_over_D"].mean()) if "L_over_D" in sub.columns else np.nan,
            "L_over_D_std": float(sub["L_over_D"].std()) if "L_over_D" in sub.columns else np.nan,
        }
        if "Cl" in sub.columns:
            row["Cl_mean"] = float(sub["Cl"].mean())
        if "Cd" in sub.columns:
            row["Cd_mean"] = float(sub["Cd"].mean())
        if "t_max" in sub.columns:
            row["t_max_mean"] = float(sub["t_max"].mean())
        if "camber_max" in sub.columns:
            row["camber_max_mean"] = float(sub["camber_max"].mean())

        for i, c in enumerate(cst_cols, start=1):
            row[f"CST_mean_{i}"] = float(sub[c].mean())
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)

    # ----- 每簇选种子：离簇中心最近的样本 -----
    centers = km.cluster_centers_
    seed_rows = []
    for cid in range(n_clusters):
        idx = np.where(labels == cid)[0]
        if len(idx) == 0:
            continue
        X_sub = X[idx]
        dists = np.linalg.norm(X_sub - centers[cid], axis=1)
        pick = idx[int(np.argmin(dists))]
        seed_rows.append(df_best.iloc[pick])

    df_seeds = pd.DataFrame(seed_rows).reset_index(drop=True)

    if verbose:
        print(f"\n种子翼型 (每簇 1 个):")
        for _, row in df_seeds.iterrows():
            fn = row.get("Filename", "?")
            cid = row.get("cluster_id", "?")
            ld = row.get("L_over_D", "?")
            print(f"  Cluster {cid}: {fn}  (L/D = {ld:.2f})" if isinstance(ld, float)
                  else f"  Cluster {cid}: {fn}")

    return df_best, df_summary, df_seeds


# ============================================================================
# Section 5: 主 API 类
# ============================================================================

class KMeansSeedSelector:
    """
    KMeans 翼型种子筛选器。

    一键完成: 数据加载 → 几何提取 → 可行性筛选 → 质量标注 → 聚类 → 种子输出

    使用示例:
        selector = KMeansSeedSelector(
            csv_path="airfoils.csv",
            dat_dir="./dat_files/",
            output_dir="./seeds_output/",
            aoa=4.0,
            n_clusters=3,
        )
        seeds_df, summary_df = selector.run()
    """

    def __init__(
        self,
        csv_path: str | Path,
        dat_dir: Optional[str | Path] = None,
        output_dir: str | Path = "./kmeans_seeds_output",
        aoa: Optional[float] = None,
        n_clusters: int = 3,
        excellent_percentile: float = 0.70,
        min_thickness: float = 0.08,
        min_cl: float = 0.3,
        min_cd: float = 0.001,
        use_cst: bool = True,
        use_thickness: bool = True,
        use_camber: bool = True,
        use_aerodynamic: bool = False,
        random_state: int = 42,
        export_dat: bool = True,
        verbose: bool = True,
    ):
        """
        参数:
            csv_path: 翼型数据 CSV 路径
                必需列: Filename, Cl, Cd
                可选列: CST Coeff 1~8 (或 CST_Coeff_1~8), AoA, t_max, camber_max
            dat_dir: .dat 翼型坐标文件夹路径。
                如果提供，将从 .dat 文件提取几何特征 (t_max, camber_max)。
                如果为 None 且 CSV 中已有 t_max/camber_max，则跳过几何提取。
            output_dir: 输出目录
            aoa: 攻角筛选。如果提供，仅保留该攻角的数据。None = 不筛选。
            n_clusters: KMeans 聚类簇数
            excellent_percentile: 优秀阈值分位数 (默认 0.70 = 前 30%)
            min_thickness: 可行性最小厚度
            min_cl: 可行性最小升力系数
            min_cd: 可行性最小阻力系数
            use_cst/thickness/camber/aerodynamic: 聚类特征开关
            random_state: 随机种子
            export_dat: 是否导出种子 .dat 文件
            verbose: 是否打印详细信息
        """
        self.csv_path = Path(csv_path)
        self.dat_dir = Path(dat_dir) if dat_dir else None
        self.output_dir = Path(output_dir)
        self.aoa = aoa
        self.n_clusters = n_clusters
        self.excellent_percentile = excellent_percentile
        self.min_thickness = min_thickness
        self.min_cl = min_cl
        self.min_cd = min_cd
        self.use_cst = use_cst
        self.use_thickness = use_thickness
        self.use_camber = use_camber
        self.use_aerodynamic = use_aerodynamic
        self.random_state = random_state
        self.export_dat = export_dat
        self.verbose = verbose

        # 中间结果
        self.df_raw: Optional[pd.DataFrame] = None
        self.df_feasible: Optional[pd.DataFrame] = None
        self.df_labeled: Optional[pd.DataFrame] = None
        self.df_good_clusters: Optional[pd.DataFrame] = None
        self.df_summary: Optional[pd.DataFrame] = None
        self.df_seeds: Optional[pd.DataFrame] = None
        self.quality_meta: Optional[Dict] = None

    # ----- 步骤 1: 加载数据 -----
    def load_data(self) -> pd.DataFrame:
        """加载 CSV 数据，可选攻角筛选 + 计算 L/D"""
        if self.verbose:
            print(f"读取 CSV: {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        if self.verbose:
            print(f"  原始样本数: {len(df)}")

        # 攻角筛选
        if self.aoa is not None:
            aoa_col = None
            for c in df.columns:
                if c.lower() == "aoa":
                    aoa_col = c
                    break
            if aoa_col:
                df = df[np.isclose(df[aoa_col].astype(float), float(self.aoa))].copy()
                if self.verbose:
                    print(f"  筛选 AoA={self.aoa}°: {len(df)} 行")
            else:
                if self.verbose:
                    print(f"  [WARN] 未找到 AoA 列，跳过攻角筛选")

        # 计算 L/D
        if "L_over_D" not in df.columns:
            if "Cl" in df.columns and "Cd" in df.columns:
                df["Cl"] = df["Cl"].astype(float)
                df["Cd"] = df["Cd"].astype(float)
                df["L_over_D"] = df["Cl"] / df["Cd"]
                if self.verbose:
                    print(f"  已计算 L_over_D = Cl / Cd")

        self.df_raw = df
        return df

    # ----- 步骤 2: 几何特征提取 -----
    def extract_geometry(self, df: pd.DataFrame) -> pd.DataFrame:
        """从 .dat 文件提取 t_max 和 camber_max"""
        df = df.copy()

        # 检查是否已有几何特征
        has_tmax = "t_max" in df.columns and df["t_max"].notna().sum() > 0
        has_camber = "camber_max" in df.columns and df["camber_max"].notna().sum() > 0

        if has_tmax and has_camber:
            if self.verbose:
                print("  几何特征已存在，跳过 .dat 文件解析")
            return df

        if self.dat_dir is None:
            if self.verbose:
                print("  [WARN] 未提供 dat_dir 且 CSV 中无几何特征，将跳过几何特征")
            if "t_max" not in df.columns:
                df["t_max"] = np.nan
            if "camber_max" not in df.columns:
                df["camber_max"] = np.nan
            return df

        if self.verbose:
            print(f"从 .dat 文件提取几何特征 (dat_dir={self.dat_dir})...")

        # 查找 Filename 列
        fn_col = None
        for c in df.columns:
            if c.lower() in ("filename", "name", "airfoil"):
                fn_col = c
                break
        if fn_col is None:
            raise ValueError("CSV 中缺少 Filename/Name/Airfoil 列，无法匹配 .dat 文件")

        filenames = df[fn_col].astype(str).unique()
        geo_cache: Dict[str, Dict[str, object]] = {}

        for i, fn in enumerate(filenames):
            if self.verbose and (i + 1) % 200 == 0:
                print(f"  进度: {i + 1}/{len(filenames)}")

            dat_path = find_dat_file(fn, self.dat_dir)
            if dat_path is None:
                geo_cache[fn] = {"t_max": np.nan, "camber_max": np.nan, "dat_path": ""}
                continue

            try:
                coords = load_dat_coords(dat_path)
                t_max_val = max_thickness(coords)
                camber_val = max_camber(coords)
                geo_cache[fn] = {
                    "t_max": t_max_val,
                    "camber_max": camber_val,
                    "dat_path": str(dat_path),
                }
            except Exception:
                geo_cache[fn] = {"t_max": np.nan, "camber_max": np.nan, "dat_path": str(dat_path)}

        df["t_max"] = df[fn_col].map(lambda x: geo_cache.get(str(x), {}).get("t_max", np.nan))
        df["camber_max"] = df[fn_col].map(lambda x: geo_cache.get(str(x), {}).get("camber_max", np.nan))
        df["dat_path"] = df[fn_col].map(lambda x: geo_cache.get(str(x), {}).get("dat_path", ""))

        n_ok = int(df["t_max"].notna().sum())
        if self.verbose:
            print(f"  成功提取 t_max: {n_ok}/{len(df)}")

        return df

    # ----- 步骤 3-5: 筛选 + 标注 + 聚类 -----
    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        执行完整流程，返回 (seeds_df, summary_df)。

        同时将结果保存到 output_dir:
          - cluster_seeds.csv: 种子翼型信息
          - cluster_summary.csv: 每簇统计
          - good_clusters.csv: 所有优秀样本 + cluster_id
          - seeds/*.dat: 种子翼型 .dat 文件 (如果 export_dat=True)
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: 加载
        df = self.load_data()

        # Step 2: 几何提取
        df = self.extract_geometry(df)

        # Step 3: 可行性筛选
        if self.verbose:
            print(f"\n可行性筛选 (t_max>={self.min_thickness}, Cl>={self.min_cl}, Cd>{self.min_cd})...")
        df_feasible = apply_feasibility_filter(
            df,
            min_thickness=self.min_thickness,
            min_cl=self.min_cl,
            min_cd=self.min_cd,
        )
        self.df_feasible = df_feasible
        if self.verbose:
            print(f"  可行样本: {len(df_feasible)}/{len(df)}")

        if len(df_feasible) == 0:
            raise ValueError("可行性筛选后无样本！请放宽阈值。")

        # Step 4: 质量标注
        if self.verbose:
            print(f"\n质量标注 (L/D >= {self.excellent_percentile*100:.0f}% 分位数 → 优秀)...")
        df_labeled, quality_meta = classify_quality(
            df_feasible,
            excellent_percentile=self.excellent_percentile,
        )
        self.df_labeled = df_labeled
        self.quality_meta = quality_meta
        if self.verbose:
            print(f"  优秀阈值 L/D* = {quality_meta['threshold_excellent_L_over_D']:.3f}")
            print(f"  优秀: {quality_meta['n_excellent']}, 合格: {quality_meta['n_qualified']}")

        # Step 5: KMeans 聚类 + 种子选择
        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f"KMeans 聚类 (k={self.n_clusters})")
            print(f"{'=' * 60}")

        df_good, df_summary, df_seeds = cluster_and_select_seeds(
            df_labeled,
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            use_cst=self.use_cst,
            use_thickness=self.use_thickness,
            use_camber=self.use_camber,
            use_aerodynamic=self.use_aerodynamic,
            verbose=self.verbose,
        )
        self.df_good_clusters = df_good
        self.df_summary = df_summary
        self.df_seeds = df_seeds

        # ----- 保存 CSV -----
        seeds_path = self.output_dir / "cluster_seeds.csv"
        df_seeds.to_csv(seeds_path, index=False, encoding="utf-8-sig")
        if self.verbose:
            print(f"\n[OK] 种子 CSV: {seeds_path}")

        summary_path = self.output_dir / "cluster_summary.csv"
        df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        if self.verbose:
            print(f"[OK] 汇总 CSV: {summary_path}")

        good_path = self.output_dir / "good_clusters.csv"
        df_good.to_csv(good_path, index=False, encoding="utf-8-sig")
        if self.verbose:
            print(f"[OK] 优秀聚类 CSV: {good_path}")

        # ----- 导出种子 .dat 文件 -----
        if self.export_dat:
            seeds_dat_dir = self.output_dir / "seeds"
            seeds_dat_dir.mkdir(parents=True, exist_ok=True)

            for _, seed in df_seeds.iterrows():
                fn = seed.get("Filename", "unknown")
                cid = seed.get("cluster_id", 0)
                safe_name = Path(str(fn)).stem
                dat_out = seeds_dat_dir / f"seed_cluster{int(cid)}_{safe_name}.dat"

                # 优先复制原始 .dat 文件
                dat_path = seed.get("dat_path", "")
                if dat_path and Path(dat_path).exists():
                    shutil.copy2(dat_path, dat_out)
                else:
                    # 从 CST 系数重构
                    try:
                        cst_cols = _resolve_cst_cols(df_seeds)
                        cst_vals = np.array([float(seed[c]) for c in cst_cols])
                        coords = cst_to_airfoil_coords(cst_vals)
                        write_dat_file(coords, dat_out, name=f"seed_cluster{int(cid)}_{safe_name}")
                    except Exception as e:
                        if self.verbose:
                            print(f"  [WARN] 无法生成 {dat_out.name}: {e}")
                        continue

            if self.verbose:
                n_dat = len(list(seeds_dat_dir.glob("*.dat")))
                print(f"[OK] 种子 .dat 文件 ({n_dat} 个): {seeds_dat_dir}")

        return df_seeds, df_summary


# ============================================================================
# Section 6: 便捷函数
# ============================================================================

def select_seeds(
    csv_path: str | Path,
    dat_dir: Optional[str | Path] = None,
    output_dir: str | Path = "./kmeans_seeds_output",
    aoa: Optional[float] = None,
    n_clusters: int = 3,
    excellent_percentile: float = 0.70,
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    便捷函数：一行代码完成种子筛选。

    参数同 KMeansSeedSelector，额外关键字参数会透传。

    返回: (seeds_df, summary_df)

    示例:
        seeds, summary = select_seeds(
            "airfoils.csv",
            dat_dir="./dat_files/",
            n_clusters=3,
        )
    """
    selector = KMeansSeedSelector(
        csv_path=csv_path,
        dat_dir=dat_dir,
        output_dir=output_dir,
        aoa=aoa,
        n_clusters=n_clusters,
        excellent_percentile=excellent_percentile,
        **kwargs,
    )
    return selector.run()


# ============================================================================
# Section 7: 命令行入口
# ============================================================================

def main():
    """命令行入口"""
    import argparse

    # Windows GBK 兼容：强制 UTF-8 输出
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="KMeans 翼型种子筛选器 — 从翼型库中选出多样化优质种子",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python kmeans_seeds.py --csv airfoils.csv --dat_dir ./dat_files/ -o ./seeds/ -k 3
  python kmeans_seeds.py --csv data.csv -k 5 --percentile 0.80 --aoa 4.0
        """,
    )

    parser.add_argument("--csv", required=True, help="翼型数据 CSV 路径")
    parser.add_argument("--dat_dir", default=None, help=".dat 翼型坐标文件夹")
    parser.add_argument("-o", "--output", default="./kmeans_seeds_output", help="输出目录")
    parser.add_argument("--aoa", type=float, default=None, help="攻角筛选 (默认不筛选)")
    parser.add_argument("-k", "--n_clusters", type=int, default=3, help="聚类簇数 (默认 3)")
    parser.add_argument("--percentile", type=float, default=0.70,
                        help="优秀 L/D 分位数 (默认 0.70 = 前 30%%)")
    parser.add_argument("--min_thickness", type=float, default=0.08, help="最小厚度")
    parser.add_argument("--min_cl", type=float, default=0.3, help="最小 Cl")
    parser.add_argument("--min_cd", type=float, default=0.001, help="最小 Cd")
    parser.add_argument("--no_cst", action="store_true", help="聚类不使用 CST 系数")
    parser.add_argument("--no_thickness", action="store_true", help="聚类不使用厚度")
    parser.add_argument("--no_camber", action="store_true", help="聚类不使用弯度")
    parser.add_argument("--use_aero", action="store_true", help="聚类使用气动参数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--no_dat", action="store_true", help="不导出 .dat 文件")
    parser.add_argument("-q", "--quiet", action="store_true", help="安静模式")

    args = parser.parse_args()

    selector = KMeansSeedSelector(
        csv_path=args.csv,
        dat_dir=args.dat_dir,
        output_dir=args.output,
        aoa=args.aoa,
        n_clusters=args.n_clusters,
        excellent_percentile=args.percentile,
        min_thickness=args.min_thickness,
        min_cl=args.min_cl,
        min_cd=args.min_cd,
        use_cst=not args.no_cst,
        use_thickness=not args.no_thickness,
        use_camber=not args.no_camber,
        use_aerodynamic=args.use_aero,
        random_state=args.seed,
        export_dat=not args.no_dat,
        verbose=not args.quiet,
    )

    seeds_df, summary_df = selector.run()

    print(f"\n{'=' * 60}")
    print("完成！种子翼型:")
    for _, row in seeds_df.iterrows():
        fn = row.get("Filename", "?")
        cid = row.get("cluster_id", "?")
        ld = row.get("L_over_D", "?")
        if isinstance(ld, float):
            print(f"  Cluster {int(cid)}: {fn}  (L/D = {ld:.4f})")
        else:
            print(f"  Cluster {int(cid)}: {fn}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

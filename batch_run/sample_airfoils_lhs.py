import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from math import comb
from scipy.stats import qmc

from config import (
    ROOT_DIR, AIRFOIL_DIR, PLOT_DIR,
    AU_BASE, AL_BASE, DZ_U, DZ_L,
    N1, N2, N_POINTS,
    N_SAMPLES, RANDOM_SEED,
    DELTA_AU, DELTA_AL,
    T_MIN_THRESHOLD, X_MARGIN_CHECK,
    ENFORCE_TMAX_NOT_SMALLER,
    N_SHOW_PLOTS, AUTO_OPEN_PLOTS
)


# =========================================================
# CST 基础函数
# =========================================================
def class_function(x, N1=0.5, N2=1.0):
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 0.0, 1.0)
    return (x ** N1) * ((1 - x) ** N2)


def bernstein_poly(i, n, x):
    return comb(n, i) * (x ** i) * ((1 - x) ** (n - i))


def shape_function(x, A):
    n = len(A) - 1
    s = np.zeros_like(x, dtype=float)
    for i in range(n + 1):
        s += A[i] * bernstein_poly(i, n, x)
    return s


def cst_surface(x, A, N1=0.5, N2=1.0, dz=0.0):
    return class_function(x, N1, N2) * shape_function(x, A) + x * dz


def rebuild_airfoil(Au, Al, dz_u, dz_l, n_points=201, N1=0.5, N2=1.0):
    beta = np.linspace(0, np.pi, n_points)
    x = 0.5 * (1 - np.cos(beta))
    yu = cst_surface(x, Au, N1=N1, N2=N2, dz=dz_u)
    yl = cst_surface(x, Al, N1=N1, N2=N2, dz=dz_l)
    return x, yu, x, yl


def write_airfoil_dat(filename, name, xu, yu, xl, yl):
    upper = np.column_stack([xu, yu])[::-1]
    lower = np.column_stack([xl, yl])[1:]
    coords = np.vstack([upper, lower])

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"{name}\n")
        for x, y in coords:
            f.write(f"{x:.8f} {y:.8f}\n")


# =========================================================
# 几何检查与厚度
# =========================================================
def calc_thickness(xu, yu, xl, yl, n_interp=500):
    x_common = np.linspace(0.0, 1.0, n_interp)
    yu_i = np.interp(x_common, xu, yu)
    yl_i = np.interp(x_common, xl, yl)
    t = yu_i - yl_i
    i = np.argmax(t)
    return {
        "x": x_common,
        "t": t,
        "t_max": t[i],
        "x_tmax": x_common[i]
    }


def is_geometry_valid(xu, yu, xl, yl, t_min_threshold=1e-5, x_margin=0.02):
    th = calc_thickness(xu, yu, xl, yl)
    x = th["x"]
    t = th["t"]

    mask = (x >= x_margin) & (x <= 1.0 - x_margin)
    t_mid = t[mask]

    if len(t_mid) == 0:
        return False

    if np.any(t_mid <= 0):
        return False

    if np.min(t_mid) < t_min_threshold:
        return False

    return True


# =========================================================
# LHS 采样
# =========================================================
def lhs_sample_cst(Au_base, Al_base, n_samples, delta_au, delta_al, seed=42):
    Au_base = np.asarray(Au_base)
    Al_base = np.asarray(Al_base)

    n_var_u = len(Au_base)
    n_var_l = len(Al_base)
    dim = n_var_u + n_var_l

    sampler = qmc.LatinHypercube(d=dim, seed=seed)
    unit_samples = sampler.random(n=n_samples)

    lower_bounds = np.concatenate([Au_base - delta_au, Al_base - delta_al])
    upper_bounds = np.concatenate([Au_base + delta_au, Al_base + delta_al])

    samples = qmc.scale(unit_samples, lower_bounds, upper_bounds)

    out = []
    for row in samples:
        Au = row[:n_var_u]
        Al = row[n_var_u:]
        out.append((Au, Al))
    return out


# =========================================================
# 图片打开
# =========================================================
def open_image_if_needed(image_path):
    if AUTO_OPEN_PLOTS:
        try:
            os.startfile(str(image_path))
        except Exception as e:
            print(f"自动打开图片失败: {image_path}")
            print(e)


# =========================================================
# 绘图
# =========================================================
def plot_all_samples(sample_geometries, baseline_geom, save_path=None):
    xu0, yu0, xl0, yl0 = baseline_geom

    plt.figure(figsize=(12, 5))

    for item in sample_geometries:
        xu, yu, xl, yl = item["xu"], item["yu"], item["xl"], item["yl"]
        status = item["status"]

        if item["is_baseline"]:
            continue

        if status == "ok":
            plt.plot(xu, yu, lw=0.8, alpha=0.35)
            plt.plot(xl, yl, lw=0.8, alpha=0.35)
        else:
            plt.plot(xu, yu, lw=1.0, alpha=0.6, linestyle="--")
            plt.plot(xl, yl, lw=1.0, alpha=0.6, linestyle="--")

    # baseline 用更粗红线压在最上面
    plt.plot(xu0, yu0, color="red", lw=3.0, zorder=10, label="Original 4412")
    plt.plot(xl0, yl0, color="red", lw=3.0, zorder=10)

    plt.axis("equal")
    plt.xlabel("x/c")
    plt.ylabel("y/c")
    plt.title("LHS sampled airfoils with original 4412")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    if save_path is not None:
        open_image_if_needed(save_path)


def plot_first_n_samples(sample_geometries, baseline_geom, n_show=12, save_path=None):
    xu0, yu0, xl0, yl0 = baseline_geom

    non_baseline_samples = [item for item in sample_geometries if not item["is_baseline"]]

    n_show = min(n_show, len(non_baseline_samples))
    ncols = 4
    nrows = int(np.ceil(n_show / ncols))

    plt.figure(figsize=(4 * ncols, 2.8 * nrows))

    for i in range(n_show):
        item = non_baseline_samples[i]
        xu, yu, xl, yl = item["xu"], item["yu"], item["xl"], item["yl"]
        status = item["status"]
        name = item["name"]

        ax = plt.subplot(nrows, ncols, i + 1)

        ax.plot(xu, yu, color="blue", lw=2.5)
        ax.plot(xl, yl, color="blue", lw=2.5)

        # baseline 轮廓红线
        ax.plot(xu0, yu0, color="red", lw=1.2)
        ax.plot(xl0, yl0, color="red", lw=1.2)

        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)
        ax.set_title(f"{name}\n{status}", fontsize=9)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    if save_path is not None:
        open_image_if_needed(save_path)


# =========================================================
# 主程序
# =========================================================
def main(au_base=None, al_base=None, dz_u=None, dz_l=None,
         seed_name=None, output_dir=None,
         au_base_label="Original Baseline"):
    """
    LHS 采样主函数。

    参数（全部可选，默认使用 config.py 中的 NACA4412 值）:
        au_base, al_base: 上下表面 CST 系数
        dz_u, dz_l: 尾缘项
        seed_name: 种子名称（用于输出文件命名）
        output_dir: 自定义输出目录（None 则用 config.ROOT_DIR）
        au_base_label: baseline 图例标签
    """
    # 使用传入值或默认值
    _au_base = AU_BASE if au_base is None else np.asarray(au_base, dtype=float)
    _al_base = AL_BASE if al_base is None else np.asarray(al_base, dtype=float)
    _dz_u = DZ_U if dz_u is None else float(dz_u)
    _dz_l = DZ_L if dz_l is None else float(dz_l)

    # 自定义输出目录
    if output_dir is not None:
        _root_dir = Path(output_dir)
        _airfoil_dir = _root_dir / "airfoils"
        _plot_dir = _root_dir / "plots"
    else:
        _root_dir = ROOT_DIR
        _airfoil_dir = AIRFOIL_DIR
        _plot_dir = PLOT_DIR

    _root_dir.mkdir(parents=True, exist_ok=True)
    _airfoil_dir.mkdir(parents=True, exist_ok=True)
    _plot_dir.mkdir(parents=True, exist_ok=True)

    # baseline 几何
    xu0, yu0, xl0, yl0 = rebuild_airfoil(
        _au_base, _al_base, _dz_u, _dz_l,
        n_points=N_POINTS, N1=N1, N2=N2
    )
    base_th = calc_thickness(xu0, yu0, xl0, yl0)
    tmax_base = base_th["t_max"]

    # 采样：sample_000 是 baseline
    lhs_samples = lhs_sample_cst(
        _au_base, _al_base,
        n_samples=N_SAMPLES,
        delta_au=DELTA_AU,
        delta_al=DELTA_AL,
        seed=RANDOM_SEED
    )
    samples = [(_au_base.copy(), _al_base.copy())] + lhs_samples

    records = []
    sample_geometries = []

    for i, (Au, Al) in enumerate(samples):
        name = f"sample_{i:03d}"
        is_baseline = (i == 0)

        xu, yu, xl, yl = rebuild_airfoil(
            Au, Al, _dz_u, _dz_l,
            n_points=N_POINTS, N1=N1, N2=N2
        )

        valid = is_geometry_valid(
            xu, yu, xl, yl,
            t_min_threshold=T_MIN_THRESHOLD,
            x_margin=X_MARGIN_CHECK
        )
        th = calc_thickness(xu, yu, xl, yl)

        status = "ok"
        if not valid:
            status = "invalid_geometry"
        elif ENFORCE_TMAX_NOT_SMALLER and (th["t_max"] < tmax_base):
            status = "thickness_too_small"

        dat_path = _airfoil_dir / f"{name}.dat"
        write_airfoil_dat(dat_path, name, xu, yu, xl, yl)

        record = {
            "name": name,
            "is_baseline": is_baseline,
            "status": status,
            "dat_file": str(dat_path),
            "t_max": th["t_max"],
            "x_tmax": th["x_tmax"],
            "dz_u": _dz_u,
            "dz_l": _dz_l,
        }

        for j in range(len(Au)):
            record[f"Au_{j}"] = Au[j]
        for j in range(len(Al)):
            record[f"Al_{j}"] = Al[j]

        records.append(record)

        sample_geometries.append({
            "name": name,
            "is_baseline": is_baseline,
            "status": status,
            "xu": xu,
            "yu": yu,
            "xl": xl,
            "yl": yl
        })

    df = pd.DataFrame(records)
    csv_path = _root_dir / "samples.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("=" * 60)
    print("LHS采样完成")
    print(f"样本表已保存: {csv_path}")
    print(df["status"].value_counts(dropna=False))
    print(f"t_max range: {df['t_max'].min():.6f} ~ {df['t_max'].max():.6f}")
    print("=" * 60)

    baseline_geom = (xu0, yu0, xl0, yl0)

    all_plot_path = _plot_dir / "all_samples_overlay.png"
    first_plot_path = _plot_dir / "first_n_samples.png"

    plot_all_samples(sample_geometries, baseline_geom, save_path=all_plot_path)
    plot_first_n_samples(sample_geometries, baseline_geom, n_show=N_SHOW_PLOTS, save_path=first_plot_path)

    print(f"总叠加图已保存: {all_plot_path}")
    print(f"前{N_SHOW_PLOTS}个样本图已保存: {first_plot_path}")

    return csv_path, _airfoil_dir


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
fit_cst_for_seed.py — 对任意 .dat 翼型文件拟合 12 维 CST 参数
==============================================================

功能：输入一个翼型 .dat 文件，输出 12 维 CST 参数 + 几何/气动基线信息。

核心函数：
    fit_cst_to_dat(dat_path) -> dict

返回字典包含：
    - au: 上表面 6 个 CST 系数 (numpy array)
    - al: 下表面 6 个 CST 系数 (numpy array)
    - dz_u: 上表面尾缘项
    - dz_l: 下表面尾缘项
    - t_max: 最大厚度比
    - cl: XFOIL 计算的升力系数（如果 run_xfoil=True）
    - cd: XFOIL 计算的阻力系数（如果 run_xfoil=True）
    - name: 翼型名称

使用方式：
    # 仅拟合 CST + 厚度（不需要 XFOIL）
    info = fit_cst_to_dat("my_airfoil.dat", run_xfoil=False)

    # 拟合 CST + 厚度 + XFOIL 算基线气动
    info = fit_cst_to_dat("my_airfoil.dat", run_xfoil=True,
                          xfoil_exe="path/to/xfoil.exe",
                          alpha=2.0, Re=6e6, Mach=0.3)

依赖：numpy, scipy, subprocess（仅 XFOIL 模式需要）
"""

from __future__ import annotations

import json
import subprocess
import sys
from math import comb
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# 用于检测坐标行的正则
_NUM_LINE_RE = __import__('re').compile(r"^\s*[-+0-9\.eE]+\s+[-+0-9\.eE]+\s*$")


def _is_coord_line(line: str) -> bool:
    """检测一行文本是否为坐标行（两个数值）"""
    return bool(_NUM_LINE_RE.match(line.strip()))


# ============================================================================
# 1. 翼型文件读取与预处理
# ============================================================================

def read_airfoil_dat(filename: str | Path) -> Tuple[str, np.ndarray]:
    """
    读取 .dat 翼型文件。兼容两种格式：
      格式A: 第一行是名称，后续行是坐标
      格式B: 所有行都是坐标（无名称头）

    返回:
        name: 翼型名称
        xy:   Nx2 坐标数组
    """
    filename = Path(filename)

    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    if len(lines) < 2:
        raise ValueError(f"翼型文件内容过少: {filename}")

    # 检测第一行是不是坐标
    first_line = lines[0].strip()
    if _is_coord_line(first_line):
        # 格式B: 无名称头，所有行都是坐标
        name = filename.stem
        coord_lines = lines
    else:
        # 格式A: 第一行是名称
        name = first_line if first_line else filename.stem
        coord_lines = lines[1:]

    coords = []
    for line in coord_lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                x = float(parts[0])
                y = float(parts[1])
                coords.append([x, y])
            except ValueError:
                continue

    if len(coords) < 10:
        raise ValueError(f"有效坐标点过少 ({len(coords)}): {filename}")

    return name, np.array(coords, dtype=float)


def normalize_x(xy: np.ndarray) -> np.ndarray:
    """将翼型 x 坐标归一化到 [0, 1]"""
    xy = xy.copy()
    x_min = np.min(xy[:, 0])
    x_max = np.max(xy[:, 0])
    chord = x_max - x_min
    if chord <= 0:
        raise ValueError("翼型 x 坐标范围无效，无法归一化。")
    xy[:, 0] = (xy[:, 0] - x_min) / chord
    return xy


def split_upper_lower(xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    分离上下表面。自动检测两种常见点序：

    格式1: TE→LE→TE — TE(x≈1)在首尾，LE(x≈0)在中间
    格式2: LE→TE→LE — LE(x≈0)在首尾，TE(x≈1)在中间

    返回:
        xu, yu: 上表面，x 从 0 到 1
        xl, yl: 下表面，x 从 0 到 1
    """
    x = xy[:, 0]
    y = xy[:, 1]
    n = len(x)

    i_le = int(np.argmin(x))   # LE: x 最小值
    i_te = int(np.argmax(x))   # TE: x 最大值

    is_le_first_or_last = (i_le == 0 or i_le == n - 1)
    is_te_in_middle = (0 < i_te < n - 1)

    if is_le_first_or_last and is_te_in_middle:
        # ---- LE→TE→LE 点序 ----
        # 在 TE 处分割
        x1, y1 = x[:i_te + 1].copy(), y[:i_te + 1].copy()
        x2, y2 = x[i_te:].copy(), y[i_te:].copy()
        # 第二段反向，使 x 从小到大
        x2, y2 = x2[::-1], y2[::-1]
        xu, yu = x1, y1
        xl, yl = x2, y2
    else:
        # ---- TE→LE→TE 点序 ----
        xu = x[:i_le + 1].copy()
        yu = y[:i_le + 1].copy()
        xl = x[i_le:].copy()
        yl = y[i_le:].copy()

        if len(xu) > 1 and xu[0] > xu[-1]:
            xu, yu = xu[::-1], yu[::-1]
        if len(xl) > 1 and xl[0] > xl[-1]:
            xl, yl = xl[::-1], yl[::-1]

    # 确保 x 在 [0, 1]
    xu = np.clip(xu, 0.0, 1.0)
    xl = np.clip(xl, 0.0, 1.0)

    # 判断上下表面：x≈0.5 处 y 更大者为上表面
    try:
        iu = int(np.argmin(np.abs(xu - 0.5)))
        il = int(np.argmin(np.abs(xl - 0.5)))
        if yu[iu] < yl[il]:
            xu, xl = xl, xu
            yu, yl = yl, yu
    except Exception:
        pass

    return xu, yu, xl, yl


# ============================================================================
# 2. CST 基础数学函数
# ============================================================================

def class_function(x: np.ndarray, N1: float = 0.5, N2: float = 1.0) -> np.ndarray:
    """CST 类函数: C(x) = x^N1 * (1-x)^N2"""
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 0.0, 1.0)
    return (x ** N1) * ((1 - x) ** N2)


def bernstein_poly(i: int, n: int, x: np.ndarray) -> np.ndarray:
    """Bernstein 多项式"""
    return comb(n, i) * (x ** i) * ((1 - x) ** (n - i))


def shape_function(x: np.ndarray, A: np.ndarray) -> np.ndarray:
    """CST 形状函数: S(x) = sum_i A_i * B_i^n(x)"""
    n = len(A) - 1
    s = np.zeros_like(x, dtype=float)
    for i in range(n + 1):
        s += A[i] * bernstein_poly(i, n, x)
    return s


def cst_surface(x: np.ndarray, A: np.ndarray,
                N1: float = 0.5, N2: float = 1.0,
                dz: float = 0.0) -> np.ndarray:
    """CST 表面: y(x) = C(x) * S(x) + x * dz"""
    C = class_function(x, N1, N2)
    S = shape_function(x, A)
    return C * S + x * dz


# ============================================================================
# 3. CST 最小二乘拟合
# ============================================================================

def fit_cst(x: np.ndarray, y: np.ndarray,
            n_coeff: int = 6,
            N1: float = 0.5, N2: float = 1.0,
            fit_dz: bool = True) -> Tuple[np.ndarray, float]:
    """
    用最小二乘法对单侧表面拟合 CST 参数。

    参数:
        x, y: 翼型表面坐标点
        n_coeff: CST 系数个数（默认 6，即 A0~A5）
        N1, N2: 类函数参数
        fit_dz: 是否拟合尾缘项 dz

    返回:
        A:  CST 系数数组 (长度 n_coeff)
        dz: 尾缘项
    """
    from scipy.optimize import least_squares

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        raise ValueError("fit_cst: x 或 y 中存在 NaN。")

    x = np.clip(x, 1e-8, 1 - 1e-8)

    if fit_dz:
        p0 = np.zeros(n_coeff + 1)

        def residual(p):
            A = p[:-1]
            dz = p[-1]
            y_fit = cst_surface(x, A, N1=N1, N2=N2, dz=dz)
            return y_fit - y

        res = least_squares(residual, p0, method="trf")
        A = res.x[:-1]
        dz = res.x[-1]
    else:
        p0 = np.zeros(n_coeff)

        def residual(p):
            A = p
            y_fit = cst_surface(x, A, N1=N1, N2=N2, dz=0.0)
            return y_fit - y

        res = least_squares(residual, p0, method="trf")
        A = res.x
        dz = 0.0

    return A, dz


def rebuild_airfoil(Au: np.ndarray, Al: np.ndarray,
                    dz_u: float, dz_l: float,
                    n_points: int = 201,
                    N1: float = 0.5, N2: float = 1.0):
    """根据上下表面 CST 参数重构翼型坐标"""
    beta = np.linspace(0, np.pi, n_points)
    x = 0.5 * (1 - np.cos(beta))
    yu = cst_surface(x, Au, N1=N1, N2=N2, dz=dz_u)
    yl = cst_surface(x, Al, N1=N1, N2=N2, dz=dz_l)
    return x, yu, x, yl


# ============================================================================
# 4. 几何厚度计算
# ============================================================================

def calc_max_thickness_from_xy(xu: np.ndarray, yu: np.ndarray,
                                xl: np.ndarray, yl: np.ndarray,
                                n_interp: int = 1000) -> Dict:
    """从分离的上下表面坐标计算最大厚度"""
    x_common = np.linspace(0.0, 1.0, n_interp)
    yu_i = np.interp(x_common, xu, yu)
    yl_i = np.interp(x_common, xl, yl)
    thickness = yu_i - yl_i
    idx = np.argmax(thickness)
    return {
        "t_max": float(thickness[idx]),
        "t_max_percent_chord": float(thickness[idx] * 100),
        "x_tmax": float(x_common[idx]),
        "x_tmax_percent_chord": float(x_common[idx] * 100),
    }


def calc_max_thickness_from_dat(dat_path: str | Path) -> float:
    """直接从 .dat 文件计算最大厚度比"""
    _, xy = read_airfoil_dat(dat_path)
    xy = normalize_x(xy)
    xu, yu, xl, yl = split_upper_lower(xy)
    result = calc_max_thickness_from_xy(xu, yu, xl, yl)
    return result["t_max"]


# ============================================================================
# 5. XFOIL 气动计算（可选）
# ============================================================================

def write_dat_for_xfoil(filename: str | Path, name: str,
                        xu: np.ndarray, yu: np.ndarray,
                        xl: np.ndarray, yl: np.ndarray) -> Path:
    """写出 XFOIL 可读的 .dat 文件 (Selig 格式)"""
    filename = Path(filename)
    upper = np.column_stack([xu, yu])[::-1]
    lower = np.column_stack([xl, yl])[1:]
    coords = np.vstack([upper, lower])

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"{name}\n")
        for x, y in coords:
            f.write(f"{x:.8f} {y:.8f}\n")

    return filename


def run_xfoil_single(xfoil_exe: str | Path,
                     airfoil_dat: str | Path,
                     alpha: float = 2.0,
                     Re: float = 6e6,
                     Mach: float = 0.3,
                     Iter: int = 200,
                     workdir: Optional[str | Path] = None) -> Dict:
    """
    用 XFOIL 计算单个翼型在指定攻角下的气动性能。

    返回:
        {
            'alpha': float, 'CL': float, 'CD': float, 'CDp': float,
            'CM': float, 'CL_CD': float, 'converged': bool
        }

    注意：需要 xfoil.exe 在工作目录下或 PATH 中。
    """
    xfoil_exe = Path(xfoil_exe)
    airfoil_dat = Path(airfoil_dat)

    if workdir is None:
        workdir = airfoil_dat.parent
    else:
        workdir = Path(workdir)

    if not xfoil_exe.exists():
        raise FileNotFoundError(f"找不到 xfoil.exe: {xfoil_exe}")

    if not airfoil_dat.exists():
        raise FileNotFoundError(f"找不到翼型文件: {airfoil_dat}")

    polar_file = workdir / "_temp_polar.txt"
    input_file = workdir / "_temp_input.in"

    # 清理旧文件
    for f in [polar_file, input_file]:
        if f.exists():
            f.unlink()

    # 写 XFOIL 命令
    with open(input_file, "w", encoding="utf-8") as f:
        f.write(f"LOAD {airfoil_dat.name}\n")
        f.write("PANE\n")
        f.write("OPER\n")
        f.write(f"VISC {Re}\n")
        f.write(f"MACH {Mach}\n")
        f.write(f"ITER {Iter}\n")
        f.write("PACC\n")
        f.write(f"{polar_file.name}\n")
        f.write("\n")
        f.write(f"ALFA {alpha}\n")
        f.write("PACC\n")
        f.write("\n")
        f.write("\n")
        f.write("QUIT\n")

    # 运行 XFOIL
    cmd = f'"{xfoil_exe.name}" < "{input_file.name}"'
    cwd = str(workdir)
    ret = subprocess.call(cmd, shell=True, cwd=cwd,
                          stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL)

    result = {
        "alpha": alpha, "CL": np.nan, "CD": np.nan, "CDp": np.nan,
        "CM": np.nan, "CL_CD": np.nan, "converged": False
    }

    if ret == 0 and polar_file.exists():
        with open(polar_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                parts = s.split()
                if len(parts) >= 7:
                    try:
                        result["alpha"] = float(parts[0])
                        result["CL"] = float(parts[1])
                        result["CD"] = float(parts[2])
                        result["CDp"] = float(parts[3])
                        result["CM"] = float(parts[4])
                        if result["CD"] != 0:
                            result["CL_CD"] = result["CL"] / result["CD"]
                        result["converged"] = True
                    except ValueError:
                        continue

    # 清理
    for f in [polar_file, input_file]:
        if f.exists():
            try:
                f.unlink()
            except OSError:
                pass

    return result


# ============================================================================
# 6. 主 API 函数
# ============================================================================

def fit_cst_to_dat(dat_path: str | Path,
                   n_coeff: int = 6,
                   N1: float = 0.5,
                   N2: float = 1.0,
                   fit_dz: bool = True,
                   run_xfoil: bool = False,
                   xfoil_exe: Optional[str | Path] = None,
                   xfoil_alpha: float = 2.0,
                   xfoil_Re: float = 6e6,
                   xfoil_Mach: float = 0.3) -> Dict:
    """
    对任意 .dat 翼型文件拟合 CST 参数并提取几何/气动基线。

    参数:
        dat_path: 翼型 .dat 文件路径
        n_coeff: CST 系数个数（6 = A0~A5）
        N1, N2: CST 类函数参数（默认 0.5, 1.0）
        fit_dz: 是否拟合尾缘 dz 项
        run_xfoil: 是否运行 XFOIL 获取气动基线
        xfoil_exe: xfoil.exe 路径（run_xfoil=True 时必须）
        xfoil_alpha: XFOIL 计算攻角
        xfoil_Re: 雷诺数
        xfoil_Mach: 马赫数

    返回:
        {
            'name': str,           # 翼型名称
            'au': np.ndarray,      # 上表面 6 个 CST 系数
            'al': np.ndarray,      # 下表面 6 个 CST 系数
            'dz_u': float,         # 上表面尾缘项
            'dz_l': float,         # 下表面尾缘项
            't_max': float,        # 最大厚度比
            'rmse_u': float,       # 上表面拟合 RMSE
            'rmse_l': float,       # 下表面拟合 RMSE
            # 以下仅 run_xfoil=True 时返回
            'cl': float | None,    # 升力系数
            'cd': float | None,    # 阻力系数
            'cl_cd': float | None, # 升阻比
            'xfoil_converged': bool, # XFOIL 是否收敛
        }
    """
    dat_path = Path(dat_path)

    # 1. 读取并归一化
    name, xy = read_airfoil_dat(dat_path)
    xy = normalize_x(xy)
    xu, yu, xl, yl = split_upper_lower(xy)

    # 2. 拟合上表面 CST
    Au, dz_u = fit_cst(xu, yu, n_coeff=n_coeff, N1=N1, N2=N2, fit_dz=fit_dz)
    yu_fit = cst_surface(xu, Au, N1=N1, N2=N2, dz=dz_u)
    rmse_u = float(np.sqrt(np.mean((yu_fit - yu) ** 2)))

    # 3. 拟合下表面 CST
    Al, dz_l = fit_cst(xl, yl, n_coeff=n_coeff, N1=N1, N2=N2, fit_dz=fit_dz)
    yl_fit = cst_surface(xl, Al, N1=N1, N2=N2, dz=dz_l)
    rmse_l = float(np.sqrt(np.mean((yl_fit - yl) ** 2)))

    # 4. 计算厚度
    thick = calc_max_thickness_from_xy(xu, yu, xl, yl)

    result = {
        "name": name,
        "dat_path": str(dat_path),
        "au": Au,
        "al": Al,
        "dz_u": float(dz_u),
        "dz_l": float(dz_l),
        "t_max": thick["t_max"],
        "x_tmax": thick["x_tmax"],
        "rmse_u": rmse_u,
        "rmse_l": rmse_l,
        "n_coeff": n_coeff,
        "N1": N1,
        "N2": N2,
        "cl": None,
        "cd": None,
        "cl_cd": None,
        "xfoil_converged": False,
    }

    # 5. 可选：XFOIL 计算气动基线
    if run_xfoil:
        if xfoil_exe is None:
            raise ValueError("run_xfoil=True 但未提供 xfoil_exe 路径")

        # 用拟合的 CST 重建翼型，导出为临时 .dat
        xru, yru, xrl, yrl = rebuild_airfoil(Au, Al, dz_u, dz_l,
                                              n_points=201, N1=N1, N2=N2)
        tmp_dat = dat_path.parent / f"_tmp_{dat_path.stem}_cst.dat"
        write_dat_for_xfoil(tmp_dat, f"{name}_CST", xru, yru, xrl, yrl)

        xfoil_result = run_xfoil_single(
            xfoil_exe=xfoil_exe,
            airfoil_dat=tmp_dat,
            alpha=xfoil_alpha,
            Re=xfoil_Re,
            Mach=xfoil_Mach,
        )

        result["cl"] = xfoil_result["CL"]
        result["cd"] = xfoil_result["CD"]
        result["cl_cd"] = xfoil_result["CL_CD"]
        result["xfoil_converged"] = xfoil_result["converged"]

        # 清理临时文件
        if tmp_dat.exists():
            try:
                tmp_dat.unlink()
            except OSError:
                pass

    return result


def save_seed_info_json(seed_info: Dict, output_path: str | Path) -> Path:
    """将种子 CST 信息保存为 JSON（方便后续加载）"""
    output_path = Path(output_path)

    # 转换 numpy 数组
    data = {}
    for k, v in seed_info.items():
        if isinstance(v, np.ndarray):
            data[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            data[k] = float(v)
        else:
            data[k] = v

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return output_path


def load_seed_info_json(json_path: str | Path) -> Dict:
    """从 JSON 加载种子 CST 信息"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 转换回 numpy 数组
    if "au" in data:
        data["au"] = np.array(data["au"], dtype=float)
    if "al" in data:
        data["al"] = np.array(data["al"], dtype=float)

    return data


# ============================================================================
# 7. 命令行入口
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="对 .dat 翼型文件拟合 12 维 CST 参数",
    )
    parser.add_argument("dat_path", help="翼型 .dat 文件路径")
    parser.add_argument("-o", "--output", default=None,
                        help="输出 JSON 路径（默认打印到屏幕）")
    parser.add_argument("--xfoil", action="store_true",
                        help="同时运行 XFOIL 获取气动基线")
    parser.add_argument("--xfoil_exe", default="xfoil.exe",
                        help="xfoil.exe 路径")
    parser.add_argument("--alpha", type=float, default=2.0,
                        help="XFOIL 攻角")
    parser.add_argument("--Re", type=float, default=6e6,
                        help="雷诺数")
    parser.add_argument("--Mach", type=float, default=0.3,
                        help="马赫数")

    args = parser.parse_args()

    info = fit_cst_to_dat(
        args.dat_path,
        run_xfoil=args.xfoil,
        xfoil_exe=args.xfoil_exe,
        xfoil_alpha=args.alpha,
        xfoil_Re=args.Re,
        xfoil_Mach=args.Mach,
    )

    print(f"\n翼型: {info['name']}")
    print(f"AU: {info['au']}")
    print(f"AL: {info['al']}")
    print(f"DZ_U: {info['dz_u']:.8f}, DZ_L: {info['dz_l']:.8f}")
    print(f"t_max: {info['t_max']:.6f}")
    print(f"拟合 RMSE - Upper: {info['rmse_u']:.2e}, Lower: {info['rmse_l']:.2e}")

    if info.get("cl") is not None:
        print(f"XFOIL: CL={info['cl']:.6f}, CD={info['cd']:.6f}, "
              f"CL/CD={info['cl_cd']:.4f}, converged={info['xfoil_converged']}")

    if args.output:
        save_seed_info_json(info, args.output)
        print(f"\n[JSON saved] {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

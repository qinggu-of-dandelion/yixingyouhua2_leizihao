# -*- coding: utf-8 -*-
"""
CST 参数拟合 + 翼型重构 + XFOIL 计算 + 最大厚度计算

使用方式：
1. 将本文件放在 4412 文件夹下；
2. 确保 4412 文件夹下有：
   - NACA4412.dat
   - xfoil.exe
3. 直接运行本 Python 文件。

最终主要输出：
- cst_xfoil_summary.xlsx    汇总 Excel，包含：
    1. CST_Parameters
    2. XFOIL_Results
    3. Thickness

同时会输出：
- NACA4412_CST.dat          CST 重构翼型文件，供 XFOIL 使用
- NACA4412_CST_fit.png      CST 拟合图
- polar_file.txt            XFOIL 原始 polar 文件
- input_file.in             XFOIL 输入命令文件
"""

from pathlib import Path
from math import comb
import subprocess

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import least_squares
from openpyxl import Workbook


# =========================================================
# 0. 用户需要修改的参数全部集中在这里
# =========================================================

# ---------- 输入文件 ----------
INPUT_AIRFOIL_NAME = "NACA4412.dat"
XFOIL_EXE_NAME = "xfoil.exe"

# ---------- CST 拟合参数 ----------
N_COEFF = 6              # CST 系数个数，6 表示 A0~A5
N1 = 0.5                 # CST 类函数参数
N2 = 1.0                 # CST 类函数参数
FIT_DZ = True            # 是否拟合尾缘项 dz
REBUILD_POINTS = 300     # CST 重构翼型点数

# ---------- XFOIL 指定工况 ----------
ALPHA_LIST = [2]         # 攻角列表，例如 [0, 2, 4, 6]
REYNOLDS = 6e6           # 雷诺数
MACH = 0.3               # 马赫数
ITER = 200               # XFOIL 最大迭代步数

# ---------- 输出文件名：全部直接保存在当前 4412 文件夹下 ----------
CST_DAT_NAME = "NACA4412_CST.dat"
CST_FIG_NAME = "NACA4412_CST_fit.png"
XFOIL_POLAR_NAME = "polar_file.txt"
SUMMARY_XLSX_NAME = "cst_xfoil_summary.xlsx"


# =========================================================
# 1. 路径设置：自动识别当前代码所在文件夹
# =========================================================

def get_script_dir():
    """
    返回当前 .py 文件所在文件夹。
    本脚本放在 4412 文件夹下，因此这里就是 4412 文件夹。
    """
    return Path(__file__).resolve().parent


def find_file_in_current_folder(filename):
    """
    优先在当前代码文件夹查找文件。
    如果找不到，再递归搜索当前文件夹下的子文件夹。
    """
    script_dir = get_script_dir()

    direct_path = script_dir / filename
    if direct_path.exists():
        return direct_path

    matches = list(script_dir.rglob(filename))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"没有找到文件：{filename}\n"
        f"当前代码所在文件夹：{script_dir}\n"
        f"请确认 {filename} 和本 Python 文件放在同一个文件夹中。"
    )


# =========================================================
# 2. Excel 输出函数
# =========================================================

def write_summary_excel(xlsx_path, sheet_rows):
    """
    输出一个 Excel 文件，包含多个 sheet。
    sheet_rows 格式：
    {
        "SheetName": [ {列名: 值, 列名: 值}, {...} ],
        ...
    }
    """
    xlsx_path = Path(xlsx_path)

    wb = Workbook()

    # 删除默认 sheet
    default_ws = wb.active
    wb.remove(default_ws)

    for sheet_name, rows in sheet_rows.items():
        ws = wb.create_sheet(title=sheet_name[:31])

        if not rows:
            continue

        headers = list(rows[0].keys())
        ws.append(headers)

        for row in rows:
            ws.append([row.get(h, "") for h in headers])

        # 简单调整列宽
        for col in ws.columns:
            max_length = 0
            col_letter = col[0].column_letter

            for cell in col:
                if cell.value is not None:
                    max_length = max(max_length, len(str(cell.value)))

            ws.column_dimensions[col_letter].width = min(max_length + 2, 30)

    wb.save(xlsx_path)


# =========================================================
# 3. 翼型读取与 CST 拟合函数
# =========================================================

def read_airfoil_dat(filename):
    """
    读取 .dat 翼型文件。

    返回：
        name: 翼型名称
        xy:   Nx2 坐标数组
    """
    filename = Path(filename)

    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    if len(lines) < 2:
        raise ValueError("翼型文件内容过少，无法读取。")

    name = lines[0].strip()
    if not name:
        name = filename.stem

    coords = []

    for line in lines[1:]:
        parts = line.strip().split()

        if len(parts) >= 2:
            try:
                x = float(parts[0])
                y = float(parts[1])
                coords.append([x, y])
            except ValueError:
                continue

    if len(coords) < 10:
        raise ValueError("有效坐标点过少，请检查 dat 文件格式。")

    xy = np.array(coords, dtype=float)
    return name, xy


def normalize_x(xy):
    """
    将翼型 x 坐标归一化到 [0, 1]。
    """
    xy = xy.copy()

    x_min = np.min(xy[:, 0])
    x_max = np.max(xy[:, 0])
    chord = x_max - x_min

    if chord <= 0:
        raise ValueError("翼型 x 坐标范围无效，无法归一化。")

    xy[:, 0] = (xy[:, 0] - x_min) / chord

    return xy


def split_upper_lower(xy):
    """
    分离上下表面。

    默认 dat 文件顺序：
        TE 上表面 -> LE -> TE 下表面

    返回：
        xu, yu: 上表面，x 从 0 到 1
        xl, yl: 下表面，x 从 0 到 1
    """
    i_le = np.argmin(xy[:, 0])

    upper = xy[:i_le + 1]
    lower = xy[i_le:]

    upper = upper[::-1]

    xu, yu = upper[:, 0], upper[:, 1]
    xl, yl = lower[:, 0], lower[:, 1]

    xu = np.clip(xu, 0.0, 1.0)
    xl = np.clip(xl, 0.0, 1.0)

    return xu, yu, xl, yl


def class_function(x, N1=0.5, N2=1.0):
    """
    CST 类函数：
        C(x) = x^N1 * (1-x)^N2
    """
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 0.0, 1.0)

    return (x ** N1) * ((1 - x) ** N2)


def bernstein_poly(i, n, x):
    """
    Bernstein 多项式。
    """
    return comb(n, i) * (x ** i) * ((1 - x) ** (n - i))


def shape_function(x, A):
    """
    CST 形状函数。
    """
    n = len(A) - 1
    s = np.zeros_like(x, dtype=float)

    for i in range(n + 1):
        s += A[i] * bernstein_poly(i, n, x)

    return s


def cst_surface(x, A, N1=0.5, N2=1.0, dz=0.0):
    """
    CST 表面表达式：
        y = C(x) * S(x) + x * dz
    """
    C = class_function(x, N1, N2)
    S = shape_function(x, A)

    return C * S + x * dz


def fit_cst(x, y, n_coeff=6, N1=0.5, N2=1.0, fit_dz=True):
    """
    拟合单侧表面 CST 参数。

    返回：
        A:  CST 系数
        dz: 尾缘项
    """
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


def rebuild_airfoil(Au, Al, dz_u, dz_l, n_points=300, N1=0.5, N2=1.0):
    """
    根据上下表面 CST 参数重构翼型。
    """
    beta = np.linspace(0, np.pi, n_points)
    x = 0.5 * (1 - np.cos(beta))

    yu = cst_surface(x, Au, N1=N1, N2=N2, dz=dz_u)
    yl = cst_surface(x, Al, N1=N1, N2=N2, dz=dz_l)

    return x, yu, x, yl


def write_airfoil_dat(filename, name, xu, yu, xl, yl):
    """
    输出 CST 重构后的 dat 文件。

    输出格式：
        TE 上表面 -> LE -> TE 下表面
    """
    filename = Path(filename)

    upper = np.column_stack([xu, yu])[::-1]
    lower = np.column_stack([xl, yl])[1:]
    coords = np.vstack([upper, lower])

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"{name}\n")

        for x, y in coords:
            f.write(f"{x:.8f} {y:.8f}\n")


def save_cst_plot(fig_path, name, xu, yu, xl, yl, xru, yru, xrl, yrl):
    """
    保存 CST 拟合图。
    """
    fig_path = Path(fig_path)

    plt.figure(figsize=(10, 4))

    plt.plot(xu, yu, "o", ms=3, label="Original Upper")
    plt.plot(xl, yl, "o", ms=3, label="Original Lower")
    plt.plot(xru, yru, "-", lw=2, label="CST Upper")
    plt.plot(xrl, yrl, "-", lw=2, label="CST Lower")

    plt.axis("equal")
    plt.xlabel("x/c")
    plt.ylabel("y/c")
    plt.title(f"CST Fitting of {name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(fig_path, dpi=300)
    plt.close()


def build_cst_parameter_rows(Au, Al, dz_u, dz_l, rmse_u, rmse_l):
    """
    构造 CST 参数表。
    这里不仅输出 CST 系数，也输出 CST 初始设置信息。
    """
    rows = []

    # 上表面 CST 系数
    for i, value in enumerate(Au):
        rows.append({
            "airfoil": INPUT_AIRFOIL_NAME,
            "surface": "upper",
            "parameter": f"A{i}",
            "value": float(value),
            "N1": N1,
            "N2": N2,
            "n_coeff": N_COEFF,
            "fit_dz": FIT_DZ,
            "rebuild_points": REBUILD_POINTS,
            "rmse": float(rmse_u),
        })

    rows.append({
        "airfoil": INPUT_AIRFOIL_NAME,
        "surface": "upper",
        "parameter": "dz",
        "value": float(dz_u),
        "N1": N1,
        "N2": N2,
        "n_coeff": N_COEFF,
        "fit_dz": FIT_DZ,
        "rebuild_points": REBUILD_POINTS,
        "rmse": float(rmse_u),
    })

    # 下表面 CST 系数
    for i, value in enumerate(Al):
        rows.append({
            "airfoil": INPUT_AIRFOIL_NAME,
            "surface": "lower",
            "parameter": f"A{i}",
            "value": float(value),
            "N1": N1,
            "N2": N2,
            "n_coeff": N_COEFF,
            "fit_dz": FIT_DZ,
            "rebuild_points": REBUILD_POINTS,
            "rmse": float(rmse_l),
        })

    rows.append({
        "airfoil": INPUT_AIRFOIL_NAME,
        "surface": "lower",
        "parameter": "dz",
        "value": float(dz_l),
        "N1": N1,
        "N2": N2,
        "n_coeff": N_COEFF,
        "fit_dz": FIT_DZ,
        "rebuild_points": REBUILD_POINTS,
        "rmse": float(rmse_l),
    })

    return rows


# =========================================================
# 4. XFOIL 计算与 polar 文件读取
# =========================================================

def run_xfoil_with_pacc(
    xfoil_exe,
    airfoil_dat,
    polar_file,
    alpha_list,
    Re=6e6,
    Mach=0.3,
    Iter=200
):
    """
    用 XFOIL + PACC 自动计算多个攻角，并输出 polar 文件。

    为避免路径问题：
    - XFOIL 在当前 4412 文件夹下运行；
    - polar_file 也直接生成在 4412 文件夹下。
    """
    xfoil_exe = Path(xfoil_exe)
    airfoil_dat = Path(airfoil_dat)
    polar_file = Path(polar_file)

    if not xfoil_exe.exists():
        raise FileNotFoundError(f"找不到 xfoil.exe: {xfoil_exe}")

    if not airfoil_dat.exists():
        raise FileNotFoundError(f"找不到翼型文件: {airfoil_dat}")

    workdir = airfoil_dat.parent
    polar_file = workdir / polar_file.name
    input_file = workdir / "input_file.in"

    if polar_file.exists():
        polar_file.unlink()

    if input_file.exists():
        input_file.unlink()

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

        for alpha in alpha_list:
            f.write(f"ALFA {alpha}\n")

        f.write("PACC\n")
        f.write("\n")
        f.write("\n")
        f.write("QUIT\n")

    cmd = f'"{xfoil_exe.name}" < "{input_file.name}"'
    ret = subprocess.call(cmd, shell=True, cwd=str(workdir))

    if ret != 0:
        raise RuntimeError(f"XFOIL 运行失败，返回码: {ret}")

    if not polar_file.exists():
        raise FileNotFoundError(
            f"XFOIL 运行后未生成 polar 文件: {polar_file}\n"
            f"请检查：\n"
            f"1. xfoil.exe 是否能正常运行；\n"
            f"2. 攻角 ALPHA_LIST 是否过大导致不收敛；\n"
            f"3. input_file.in 中的 XFOIL 命令是否正确。"
        )

    return polar_file


def read_polar_file_as_rows(polar_file):
    """
    读取 XFOIL polar 文件，输出为表格行。
    """
    polar_file = Path(polar_file)

    if not polar_file.exists():
        raise FileNotFoundError(f"找不到 polar 文件: {polar_file}")

    rows = []

    with open(polar_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()

            if not s:
                continue

            parts = s.split()

            if len(parts) >= 7:
                try:
                    alpha = float(parts[0])
                    cl = float(parts[1])
                    cd = float(parts[2])
                    cdp = float(parts[3])
                    cm = float(parts[4])
                    top_xtr = float(parts[5])
                    bot_xtr = float(parts[6])

                    if cd != 0:
                        cl_cd = cl / cd
                    else:
                        cl_cd = np.nan

                    rows.append({
                        "airfoil": CST_DAT_NAME,
                        "alpha_deg": alpha,
                        "CL": cl,
                        "CD": cd,
                        "CDp": cdp,
                        "CM": cm,
                        "Top_Xtr": top_xtr,
                        "Bot_Xtr": bot_xtr,
                        "CL_CD": cl_cd,
                        "Re": float(REYNOLDS),
                        "Mach": float(MACH),
                        "Iter": int(ITER),
                    })

                except ValueError:
                    continue

    if not rows:
        raise ValueError(
            "polar 文件中没有读到有效数据。\n"
            "可能原因：XFOIL 未收敛、攻角过大、翼型格式有问题，或 xfoil.exe 没有正常运行。"
        )

    return rows


# =========================================================
# 5. 最大厚度计算
# =========================================================

def read_airfoil_coords(dat_file):
    """
    读取翼型坐标。
    """
    pts = []

    with open(dat_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) < 2:
                continue

            try:
                x = float(parts[0])
                y = float(parts[1])
                pts.append([x, y])
            except ValueError:
                continue

    pts = np.array(pts, dtype=float)

    if len(pts) < 10:
        raise ValueError("翼型坐标点太少，读取失败。")

    return pts


def calc_max_thickness(dat_file):
    """
    计算最大厚度及其位置。
    默认 dat 顺序：
        TE 上表面 -> LE -> TE 下表面
    """
    pts = read_airfoil_coords(dat_file)

    pts = normalize_x(pts)

    i_le = np.argmin(pts[:, 0])

    upper = pts[:i_le + 1][::-1]
    lower = pts[i_le:]

    xu, yu = upper[:, 0], upper[:, 1]
    xl, yl = lower[:, 0], lower[:, 1]

    x_common = np.linspace(0.0, 1.0, 1000)

    yu_i = np.interp(x_common, xu, yu)
    yl_i = np.interp(x_common, xl, yl)

    thickness = yu_i - yl_i
    idx = np.argmax(thickness)

    return {
        "airfoil": Path(dat_file).name,
        "t_max": float(thickness[idx]),
        "t_max_percent_chord": float(thickness[idx] * 100),
        "x_tmax": float(x_common[idx]),
        "x_tmax_percent_chord": float(x_common[idx] * 100),
    }


# =========================================================
# 6. 主程序
# =========================================================

def main():
    script_dir = get_script_dir()

    input_airfoil = find_file_in_current_folder(INPUT_AIRFOIL_NAME)
    xfoil_exe = find_file_in_current_folder(XFOIL_EXE_NAME)

    output_dat = script_dir / CST_DAT_NAME
    output_fig = script_dir / CST_FIG_NAME
    polar_file = script_dir / XFOIL_POLAR_NAME
    summary_xlsx = script_dir / SUMMARY_XLSX_NAME

    print("=" * 70)
    print("自动识别路径")
    print(f"当前代码目录：{script_dir}")
    print(f"输入翼型文件：{input_airfoil}")
    print(f"XFOIL 程序：{xfoil_exe}")
    print(f"Excel 输出文件：{summary_xlsx}")
    print("=" * 70)

    # ---------- 1. CST 拟合 ----------
    name, xy = read_airfoil_dat(input_airfoil)
    xy = normalize_x(xy)

    xu, yu, xl, yl = split_upper_lower(xy)

    Au, dz_u = fit_cst(
        xu,
        yu,
        n_coeff=N_COEFF,
        N1=N1,
        N2=N2,
        fit_dz=FIT_DZ,
    )

    Al, dz_l = fit_cst(
        xl,
        yl,
        n_coeff=N_COEFF,
        N1=N1,
        N2=N2,
        fit_dz=FIT_DZ,
    )

    xru, yru, xrl, yrl = rebuild_airfoil(
        Au,
        Al,
        dz_u,
        dz_l,
        n_points=REBUILD_POINTS,
        N1=N1,
        N2=N2,
    )

    yu_fit = cst_surface(xu, Au, N1=N1, N2=N2, dz=dz_u)
    yl_fit = cst_surface(xl, Al, N1=N1, N2=N2, dz=dz_l)

    rmse_u = np.sqrt(np.mean((yu_fit - yu) ** 2))
    rmse_l = np.sqrt(np.mean((yl_fit - yl) ** 2))

    write_airfoil_dat(
        output_dat,
        f"{name}_CST",
        xru,
        yru,
        xrl,
        yrl,
    )

    save_cst_plot(
        output_fig,
        name,
        xu,
        yu,
        xl,
        yl,
        xru,
        yru,
        xrl,
        yrl,
    )

    cst_rows = build_cst_parameter_rows(
        Au,
        Al,
        dz_u,
        dz_l,
        rmse_u,
        rmse_l,
    )

    # ---------- 2. XFOIL 计算 ----------
    polar_path = run_xfoil_with_pacc(
        xfoil_exe=xfoil_exe,
        airfoil_dat=output_dat,
        polar_file=polar_file,
        alpha_list=ALPHA_LIST,
        Re=REYNOLDS,
        Mach=MACH,
        Iter=ITER,
    )

    xfoil_rows = read_polar_file_as_rows(polar_path)

    # ---------- 3. 最大厚度计算 ----------
    thickness_result = calc_max_thickness(output_dat)
    thickness_rows = [thickness_result]

    # ---------- 4. 只输出一个 Excel 汇总文件 ----------
    write_summary_excel(
        summary_xlsx,
        {
            "CST_Parameters": cst_rows,
            "XFOIL_Results": xfoil_rows,
            "Thickness": thickness_rows,
        },
    )

    # ---------- 5. 打印结果摘要 ----------
    print("\n" + "=" * 70)
    print("运行完成")
    print("-" * 70)

    print(f"CST 重构翼型文件：{output_dat}")
    print(f"CST 拟合图片：{output_fig}")
    print(f"XFOIL polar 文件：{polar_path}")
    print(f"汇总 Excel 文件：{summary_xlsx}")

    print("-" * 70)
    print(f"Upper RMSE = {rmse_u:.8e}")
    print(f"Lower RMSE = {rmse_l:.8e}")

    print("-" * 70)
    print("最大厚度结果：")
    print(f"最大厚度 t_max = {thickness_result['t_max']:.6f}")
    print(f"最大厚度百分比 = {thickness_result['t_max_percent_chord']:.3f}%")
    print(f"最大厚度位置 x/c = {thickness_result['x_tmax']:.6f}")
    print(f"最大厚度位置百分比 = {thickness_result['x_tmax_percent_chord']:.3f}%")

    print("-" * 70)
    print("XFOIL 计算结果：")

    for row in xfoil_rows:
        print(
            f"alpha = {row['alpha_deg']:.3f}, "
            f"CL = {row['CL']:.6f}, "
            f"CD = {row['CD']:.6f}, "
            f"CM = {row['CM']:.6f}, "
            f"CL/CD = {row['CL_CD']:.6f}"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()
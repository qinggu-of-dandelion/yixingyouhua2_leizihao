import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


# =========================
# 注意改工况参数！！！注意文件路径！！！
# =========================
# 当前代码文件假设位于 youhua 文件夹中
# =========================================================
# 当前代码所在目录：yixingyouhua/youhua
ROOT_DIR = Path(__file__).resolve().parent
# 项目总目录：yixingyouhua
PROJECT_DIR = ROOT_DIR.parent
# 优化结果目录：yixingyouhua/youhua/optimization_results
OPT_RESULT_DIR = ROOT_DIR / "optimization_results"
# 1. 优化汇总 Excel
RESULT_EXCEL = OPT_RESULT_DIR / "data" / "airfoil_opt_svr_summary.xlsx"
# 2. 优化翼型文件夹
AIRFOIL_DIR = OPT_RESULT_DIR / "airfoils"
# 3. 基准翼型 NACA4412
BASELINE_DAT = PROJECT_DIR / "4412" / "NACA4412.dat"
# 4. XFOIL 程序
XFOIL_EXE = AIRFOIL_DIR / "xfoil.exe"
# 5. XFOIL 校核结果保存目录
SAVE_DIR = OPT_RESULT_DIR / "xfoil_verify"
# 6. 极曲线保存目录
POLAR_DIR = SAVE_DIR / "polars"
# 自动创建需要保存结果的文件夹
SAVE_DIR.mkdir(parents=True, exist_ok=True)
POLAR_DIR.mkdir(parents=True, exist_ok=True)

CASE_NAME = "airfoil_opt"
MODEL_TYPE = "kriging"  #可以切换成 nn 或 svr（看选择了什么代理模型）

RE = 6e6
MACH = 0.3
ITER = 200
ALPHA_LIST = [2.0]
TARGET_ALPHA = 2.0


# =========================
# 基础函数
# =========================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def read_polar_file(polar_file):
    rows = []
    with open(polar_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 7:
                try:
                    rows.append([float(v) for v in parts[:7]])
                except ValueError:
                    pass

    if not rows:
        return None

    arr = np.array(rows)
    return {
        "alpha": arr[:, 0],
        "cl": arr[:, 1],
        "cd": arr[:, 2],
        "cm": arr[:, 4],
    }


def get_result_at_alpha(polar_data, target_alpha):
    idx = np.argmin(np.abs(polar_data["alpha"] - target_alpha))
    return {
        "alpha": float(polar_data["alpha"][idx]),
        "cl": float(polar_data["cl"][idx]),
        "cd": float(polar_data["cd"][idx]),
        "cm": float(polar_data["cm"][idx]),
    }


def read_airfoil_coords(dat_file):
    pts = []
    with open(dat_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                pts.append([float(parts[0]), float(parts[1])])
            except ValueError:
                pass
    pts = np.array(pts)
    if len(pts) < 10:
        raise ValueError(f"翼型坐标点太少: {dat_file}")
    return pts


def split_upper_lower(pts):
    i_le = np.argmin(pts[:, 0])
    upper = pts[:i_le + 1][::-1]
    lower = pts[i_le:]
    return upper, lower


def calc_max_thickness(dat_file):
    pts = read_airfoil_coords(dat_file)
    upper, lower = split_upper_lower(pts)

    xu, yu = upper[:, 0], upper[:, 1]
    xl, yl = lower[:, 0], lower[:, 1]

    x_common = np.linspace(0.0, 1.0, 500)
    yu_i = np.interp(x_common, xu, yu)
    yl_i = np.interp(x_common, xl, yl)
    t = yu_i - yl_i

    idx = np.argmax(t)
    return float(t[idx])


def run_xfoil(xfoil_exe, workdir, airfoil_name, polar_name, alpha_list, Re, Mach, Iter):
    workdir = Path(workdir)
    input_file = workdir / "input_file.in"
    polar_file = workdir / polar_name

    if polar_file.exists():
        polar_file.unlink()

    with open(input_file, "w", encoding="utf-8") as f:
        f.write(f"LOAD {airfoil_name}\n")
        f.write("PANE\n")
        f.write("OPER\n")
        f.write(f"VISC {Re}\n")
        f.write(f"MACH {Mach}\n")
        f.write("PACC\n")
        f.write(f"{polar_name}\n")
        f.write("\n")
        f.write(f"ITER {Iter}\n")
        for alpha in alpha_list:
            f.write(f"ALFA {alpha}\n")
        f.write("PACC\n\n\nQUIT\n")

    cmd = f'"{xfoil_exe}" < "{input_file.name}"'
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=str(workdir),
        capture_output=True,
        text=True
    )

    if result.returncode != 0 or not polar_file.exists():
        return None
    return polar_file


def verify_one(name, dat_path, workdir):
    dat_path = Path(dat_path)
    if not dat_path.exists():
        return {"name": name, "status": "dat_not_found"}

    polar_name = f"{name}_polar.txt"
    polar_path = run_xfoil(
        xfoil_exe=XFOIL_EXE,
        workdir=workdir,
        airfoil_name=dat_path.name,
        polar_name=polar_name,
        alpha_list=ALPHA_LIST,
        Re=RE,
        Mach=MACH,
        Iter=ITER
    )

    if polar_path is None:
        return {"name": name, "status": "xfoil_failed"}

    polar_target = Path(POLAR_DIR) / polar_name
    if polar_target.exists():
        polar_target.unlink()
    polar_path.replace(polar_target)

    polar_data = read_polar_file(polar_target)
    if polar_data is None:
        return {"name": name, "status": "polar_parse_failed"}

    aero = get_result_at_alpha(polar_data, TARGET_ALPHA)
    t_max = calc_max_thickness(dat_path)

    return {
        "name": name,
        "status": "ok",
        "cl": aero["cl"],
        "cd": aero["cd"],
        "cm": aero["cm"],
        "t_max": t_max,
        "dat_file": str(dat_path),
        "polar_file": str(polar_target),
    }


def build_opt_dat(algorithm):
    file_tag = f"{CASE_NAME}_{MODEL_TYPE}_{algorithm}"
    return os.path.join(AIRFOIL_DIR, f"{file_tag}_optimized_airfoil.dat")

def add_error_analysis(df):
    """
    基于 Excel 预测值和 XFOIL 验证值，计算绝对误差和相对误差
    """
    df = df.copy()

    # 绝对误差
    df["cl_error"] = df["xfoil_cl"] - df["excel_cl"]
    df["cd_error"] = df["xfoil_cd"] - df["excel_cd"]
    df["t_max_error"] = df["xfoil_t_max"] - df["excel_t_max"]

    # 相对误差（百分比）
    def safe_pct_error(real_col, pred_col):
        out = []
        for real_v, pred_v in zip(df[real_col], df[pred_col]):
            if pd.isna(real_v) or pd.isna(pred_v) or pred_v == 0:
                out.append(np.nan)
            else:
                out.append((real_v - pred_v) / pred_v * 100.0)
        return out

    df["cl_error_pct"] = safe_pct_error("xfoil_cl", "excel_cl")
    df["cd_error_pct"] = safe_pct_error("xfoil_cd", "excel_cd")
    df["t_max_error_pct"] = safe_pct_error("xfoil_t_max", "excel_t_max")

    return df

# =========================
# 主程序
# =========================

def main():
    ensure_dir(SAVE_DIR)
    ensure_dir(POLAR_DIR)

    df = pd.read_excel(RESULT_EXCEL, sheet_name="summary")
    if "algorithm" not in df.columns:
        raise ValueError("summary sheet 里缺少 algorithm 列")

    results = []

    # baseline
    baseline = verify_one("baseline", BASELINE_DAT, Path(BASELINE_DAT).parent)
    results.append({
        "type": "baseline",
        "algorithm": "baseline",
        "excel_cl": df["baseline_cl"].iloc[0] if "baseline_cl" in df.columns else np.nan,
        "excel_cd": df["baseline_cd"].iloc[0] if "baseline_cd" in df.columns else np.nan,
        "excel_t_max": df["baseline_t_max"].iloc[0] if "baseline_t_max" in df.columns else np.nan,
        "xfoil_cl": baseline.get("cl", np.nan),
        "xfoil_cd": baseline.get("cd", np.nan),
        "xfoil_t_max": baseline.get("t_max", np.nan),
        #"status": baseline["status"],
        #"dat_file": baseline.get("dat_file", ""),
        #"polar_file": baseline.get("polar_file", "")
    })

    # optimized
    for _, row in df.iterrows():
        algo = row["algorithm"]
        dat_path = build_opt_dat(algo)
        out = verify_one(f"{CASE_NAME}_{MODEL_TYPE}_{algo}", dat_path, AIRFOIL_DIR)

        results.append({
            "type": "optimized",
            "algorithm": algo,
            "excel_cl": row["opt_cl"] if "opt_cl" in row else np.nan,
            "excel_cd": row["opt_cd"] if "opt_cd" in row else np.nan,
            "excel_t_max": row["opt_t_max"] if "opt_t_max" in row else np.nan,
            "xfoil_cl": out.get("cl", np.nan),
            "xfoil_cd": out.get("cd", np.nan),
            "xfoil_t_max": out.get("t_max", np.nan),
            #"status": out["status"],
            #"dat_file": out.get("dat_file", ""),
            #"polar_file": out.get("polar_file", "")
        })

        print(f"{algo}: status={out['status']}, cl={out.get('cl', np.nan):.4f}, cd={out.get('cd', np.nan):.5f}")

    out_df = pd.DataFrame(results)

    # 加入误差分析
    out_df = add_error_analysis(out_df)

    csv_path = os.path.join(SAVE_DIR, f"{CASE_NAME}_{MODEL_TYPE}_xfoil_verify.csv")
    xlsx_path = os.path.join(SAVE_DIR, f"{CASE_NAME}_{MODEL_TYPE}_xfoil_verify.xlsx")

    out_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        # 完整验证结果
        out_df.to_excel(writer, sheet_name="xfoil_verify", index=False)

        # 只保留优化结果行，方便看误差
        opt_df = out_df[out_df["type"] == "optimized"].copy()
        opt_df.to_excel(writer, sheet_name="optimized_only", index=False)

        # 误差汇总表
        summary_rows = []
        for col in ["cl_error", "cd_error", "t_max_error",
                    "cl_error_pct", "cd_error_pct", "t_max_error_pct"]:
            summary_rows.append({
                "metric": col,
                "mean": opt_df[col].mean(),
                "std": opt_df[col].std(),
                "min": opt_df[col].min(),
                "max": opt_df[col].max()
            })

        error_summary_df = pd.DataFrame(summary_rows)
        error_summary_df.to_excel(writer, sheet_name="error_summary", index=False)

    print("验证完成：")
    print(csv_path)
    print(xlsx_path)


if __name__ == "__main__":
    main()
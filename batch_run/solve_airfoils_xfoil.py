from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os
import shutil
import subprocess
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path

from config import (
    ROOT_DIR, AIRFOIL_DIR, POLAR_DIR,
    XFOIL_EXE,
    RE, MACH, ITER,
    ALPHA_LIST, TARGET_ALPHA
)


# =========================================================
# 并行设置
# =========================================================
# XFOIL 本身不能用 GPU 加速，这里使用 CPU 多进程并行。
# 如果并行出现不稳定，把 PARALLEL 改成 False 即可回到原串行逻辑。
PARALLEL = True

# 建议不要开满全部 CPU 核心；XFOIL 会频繁读写文件，太多进程会抢磁盘。
MAX_WORKERS = 8

# 并行时每个样本使用独立临时目录，避免 input_file.in / polar 文件互相覆盖。
TMP_DIR = ROOT_DIR / "xfoil_tmp"


def read_polar_file(polar_file):
    rows = []
    with open(polar_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) >= 7:
                try:
                    rows.append([float(v) for v in parts[:7]])
                except ValueError:
                    continue

    if len(rows) == 0:
        return None

    arr = np.array(rows)
    return {
        "alpha": arr[:, 0],
        "cl": arr[:, 1],
        "cd": arr[:, 2],
        "cdp": arr[:, 3],
        "cm": arr[:, 4],
        "top_xtr": arr[:, 5],
        "bot_xtr": arr[:, 6],
    }


def get_result_at_alpha(polar_data, target_alpha):
    idx = np.argmin(np.abs(polar_data["alpha"] - target_alpha))
    return {
        "alpha": polar_data["alpha"][idx],
        "cl": polar_data["cl"][idx],
        "cd": polar_data["cd"][idx],
        "cm": polar_data["cm"][idx],
        "cdp": polar_data["cdp"][idx],
        "top_xtr": polar_data["top_xtr"][idx],
        "bot_xtr": polar_data["bot_xtr"][idx],
    }


def read_airfoil_coords(dat_file):
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
    thickness = yu_i - yl_i

    idx = np.argmax(thickness)
    return thickness[idx], x_common[idx]


def run_xfoil_with_pacc(
    xfoil_exe,
    workdir,
    airfoil_name,
    polar_name,
    alpha_list,
    Re=6e6,
    Mach=0.3,
    Iter=200
):
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

        f.write("PACC\n")
        f.write("\n")
        f.write("\n")
        f.write("QUIT\n")

    cmd = f'"{xfoil_exe}" < "{input_file.name}"'
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=str(workdir),
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("XFOIL return code:", result.returncode)
        print("XFOIL stdout:")
        print(result.stdout)
        print("XFOIL stderr:")
        print(result.stderr)
        return None

    if not polar_file.exists():
        print(f"未生成 polar 文件: {polar_file}")
        print("XFOIL stdout:")
        print(result.stdout)
        print("XFOIL stderr:")
        print(result.stderr)
        return None

    return polar_file


def solve_one_sample(row, use_temp_workdir=False):
    """
    求解一个样本。

    use_temp_workdir=True 时，每个样本复制 dat 到独立临时目录运行 XFOIL，
    适合多进程并行，避免共享 AIRFOIL_DIR 下的 input_file.in。
    """
    order = row.get("__order", -1)
    name = row["name"]
    dat_path = Path(row["dat_file"])
    polar_name = f"{name}_polar.txt"

    try:
        if use_temp_workdir:
            TMP_DIR.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=f"{name}_", dir=str(TMP_DIR)) as tmp:
                workdir = Path(tmp)
                local_dat = workdir / dat_path.name
                shutil.copy2(dat_path, local_dat)

                polar_path = run_xfoil_with_pacc(
                    xfoil_exe=XFOIL_EXE,
                    workdir=workdir,
                    airfoil_name=local_dat.name,
                    polar_name=polar_name,
                    alpha_list=ALPHA_LIST,
                    Re=RE,
                    Mach=MACH,
                    Iter=ITER
                )

                if polar_path is None:
                    return {
                        "__order": order,
                        "name": name,
                        "is_baseline": row.get("is_baseline", False),
                        "status": "xfoil_failed"
                    }

                polar_target = POLAR_DIR / polar_name
                if polar_target.exists():
                    polar_target.unlink()
                shutil.move(str(polar_path), str(polar_target))
        else:
            polar_path = run_xfoil_with_pacc(
                xfoil_exe=XFOIL_EXE,
                workdir=AIRFOIL_DIR,
                airfoil_name=dat_path.name,
                polar_name=polar_name,
                alpha_list=ALPHA_LIST,
                Re=RE,
                Mach=MACH,
                Iter=ITER
            )

            if polar_path is None:
                return {
                    "__order": order,
                    "name": name,
                    "is_baseline": row.get("is_baseline", False),
                    "status": "xfoil_failed"
                }

            polar_target = POLAR_DIR / polar_name
            if polar_target.exists():
                polar_target.unlink()
            polar_path.replace(polar_target)

        polar_data = read_polar_file(polar_target)
        if polar_data is None:
            return {
                "__order": order,
                "name": name,
                "is_baseline": row.get("is_baseline", False),
                "status": "polar_parse_failed"
            }

        aero = get_result_at_alpha(polar_data, TARGET_ALPHA)
        t_max, x_tmax = calc_max_thickness(dat_path)

        result_row = {
            "__order": order,
            "name": name,
            "is_baseline": row.get("is_baseline", False),
            "status": "ok",
            "alpha": aero["alpha"],
            "cl": aero["cl"],
            "cd": aero["cd"],
            "cm": aero["cm"],
            "ld": aero["cl"] / aero["cd"] if aero["cd"] != 0 else np.nan,
            "t_max": t_max,
            "x_tmax": x_tmax,
            "polar_file": str(polar_target),
            "dat_file": str(dat_path),
            "dz_u": row.get("dz_u", np.nan),
            "dz_l": row.get("dz_l", np.nan),
        }

        for j in range(6):
            result_row[f"Au_{j}"] = row[f"Au_{j}"]
        for j in range(6):
            result_row[f"Al_{j}"] = row[f"Al_{j}"]

        return result_row

    except Exception as e:
        return {
            "__order": order,
            "name": name,
            "is_baseline": row.get("is_baseline", False),
            "status": "worker_failed",
            "error": str(e)
        }


def print_sample_result(result, prefix=""):
    name = result["name"]
    status = result["status"]
    if status == "ok":
        print(
            f"{prefix}{name}: "
            f"Cl={result['cl']:.4f}, "
            f"Cd={result['cd']:.5f}, "
            f"Cm={result['cm']:.4f}, "
            f"t_max={result['t_max']:.5f}"
        )
    else:
        extra = f" ({result.get('error')})" if result.get("error") else ""
        print(f"{prefix}{name}: {status}{extra}")


def run_serial(df_ok):
    results = []
    for i, (_, row) in enumerate(df_ok.iterrows(), start=1):
        row_dict = row.to_dict()
        row_dict["__order"] = i - 1
        result = solve_one_sample(row_dict, use_temp_workdir=False)
        results.append(result)
        print_sample_result(result)
    return results


def run_parallel(df_ok):
    rows = df_ok.reset_index(drop=True).to_dict("records")
    for i, row in enumerate(rows):
        row["__order"] = i

    max_workers = min(MAX_WORKERS, len(rows))
    print(f"并行模式：MAX_WORKERS = {max_workers}")

    results = []
    completed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(solve_one_sample, row, True): row["name"]
            for row in rows
        }

        for future in as_completed(future_map):
            completed += 1
            result = future.result()
            results.append(result)
            print_sample_result(result, prefix=f"[{completed}/{len(rows)}] ")

    results.sort(key=lambda item: item.get("__order", 10**12))
    return results


def main(samples_csv=None, output_csv=None, working_dir=None):
    """
    XFOIL 批量求解主函数。

    参数（全部可选，默认使用 config.py 中的路径）:
        samples_csv:  LHS 采样结果 CSV 路径（默认 ROOT_DIR/samples.csv）
        output_csv:   输出 surrogate_dataset.csv 路径
        working_dir:  工作目录（包含 airfoils/ 子目录和 xfoil.exe）
                      设置后 ROOT_DIR/AIRFOIL_DIR/POLAR_DIR 均指向此目录
    """
    global ROOT_DIR, AIRFOIL_DIR, POLAR_DIR, TMP_DIR
    _orig_root = ROOT_DIR
    _orig_airfoil = AIRFOIL_DIR
    _orig_polar = POLAR_DIR
    _orig_tmp = TMP_DIR

    if working_dir is not None:
        wd = Path(working_dir)
        ROOT_DIR = wd
        AIRFOIL_DIR = wd / "airfoils"
        POLAR_DIR = wd / "polars"
        TMP_DIR = wd / "xfoil_tmp"

    try:
        ROOT_DIR.mkdir(parents=True, exist_ok=True)
        POLAR_DIR.mkdir(parents=True, exist_ok=True)

        if samples_csv is None:
            samples_csv = ROOT_DIR / "samples.csv"
        else:
            samples_csv = Path(samples_csv)

        if not samples_csv.exists():
            raise FileNotFoundError(f"找不到样本文件: {samples_csv}")

        df = pd.read_csv(samples_csv)

        print("samples.csv 状态统计:")
        print(df["status"].value_counts(dropna=False))

        df_ok = df[df["status"] == "ok"].copy()
        print("可求解样本数:", len(df_ok))

        max_cases_env = os.environ.get("XFOIL_MAX_CASES")
        if max_cases_env:
            max_cases = int(max_cases_env)
            df_ok = df_ok.head(max_cases).copy()
            print(f"调试模式：仅求解前 {max_cases} 个 ok 样本")

        if len(df_ok) == 0:
            print("没有可求解样本。")
            return

        if PARALLEL:
            results = run_parallel(df_ok)
        else:
            results = run_serial(df_ok)

        if output_csv is None:
            if max_cases_env:
                output_csv = ROOT_DIR / f"surrogate_dataset_debug_{max_cases_env}.csv"
            else:
                output_csv = ROOT_DIR / "surrogate_dataset.csv"
        else:
            output_csv = Path(output_csv)

        result_df = pd.DataFrame(results)
        result_df = result_df.drop(columns=["__order"], errors="ignore")
        result_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

        print("=" * 60)
        print(f"批量求解完成，代理模型数据集已保存: {output_csv}")
        print("=" * 60)

        df_ok2 = result_df[result_df["status"] == "ok"].copy()
        if len(df_ok2) > 0:
            print("按 Cd 从小到大排序前10个：")
            print(df_ok2.sort_values("cd")[["name", "is_baseline", "cl", "cd", "cm", "t_max"]].head(10))

        return output_csv
    finally:
        ROOT_DIR = _orig_root
        AIRFOIL_DIR = _orig_airfoil
        POLAR_DIR = _orig_polar
        TMP_DIR = _orig_tmp


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

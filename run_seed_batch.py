# -*- coding: utf-8 -*-
"""
run_seed_batch.py — 批量翼型优化脚本
======================================

输入一个包含 .dat 翼型文件的文件夹，自动对每个翼型跑完整优化流水线。

完整流水线（每个翼型独立运行）:
  Step 1:  CST 拟合            → seed_info.json
  Step 2:  XFOIL 基线计算      → seed_info.json (追加 cl, cd)
  Step 3:  LHS 采样            → samples.csv + airfoils/
  Step 4:  XFOIL 批量求解       → surrogate_dataset.csv
  Step 5:  代理模型训练         → surrogate_saved_models/
  Step 6:  优化算法             → optimization_results/

使用方式:
  # 完整流水线（需要 xfoil.exe）
  python run_seed_batch.py --dat_dir ./multi_seed_output/seeds/ --xfoil_exe ./batch_run/airfoils/xfoil.exe

  # 快速模式：只做 CST拟合+LHS采样（不需要 XFOIL）
  python run_seed_batch.py --dat_dir ./seeds/ --stop_after sampling

  # 指定优化算法和代理模型
  python run_seed_batch.py --dat_dir ./seeds/ --algo GA --model nn

依赖：本项目的 fit_cst_for_seed.py, batch_run/*, surrogate/*, youhua/*
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "batch_run"))
sys.path.insert(0, str(PROJECT_ROOT / "youhua"))
sys.path.insert(0, str(PROJECT_ROOT / "surrogate"))

from fit_cst_for_seed import (
    fit_cst_to_dat, save_seed_info_json, load_seed_info_json,
    rebuild_airfoil, write_dat_for_xfoil, run_xfoil_single,
)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_name(s: str) -> str:
    import re
    s = s.strip()
    s = re.sub(r'[\\/:*?"<>|]', '_', s)
    s = re.sub(r'\s+', '_', s)
    return s[:60]


# ============================================================================
# 单翼型优化流水线
# ============================================================================

class SingleAirfoilRunner:
    """对单个 .dat 翼型文件跑完整优化流水线"""

    def __init__(
        self,
        dat_path: str | Path,
        output_dir: str | Path,
        xfoil_exe: Optional[str | Path] = None,
        # LHS 采样参数
        n_samples: int = 1500,
        delta_au: float = 0.065,
        delta_al: float = 0.065,
        # XFOIL 参数
        xfoil_alpha: float = 2.0,
        xfoil_Re: float = 6e6,
        xfoil_Mach: float = 0.3,
        xfoil_Iter: int = 200,
        # 优化参数
        algorithm: str = "RL",
        model_type: str = "kriging",
        # 控制
        stop_after: str = "full",  # "cst" / "sampling" / "xfoil" / "surrogate" / "full"
        verbose: bool = True,
    ):
        self.dat_path = Path(dat_path)
        self.output_dir = Path(output_dir)
        self.xfoil_exe = Path(xfoil_exe) if xfoil_exe else None
        self.n_samples = n_samples
        self.delta_au = delta_au
        self.delta_al = delta_al
        self.xfoil_alpha = xfoil_alpha
        self.xfoil_Re = xfoil_Re
        self.xfoil_Mach = xfoil_Mach
        self.xfoil_Iter = xfoil_Iter
        self.algorithm = algorithm
        self.model_type = model_type
        self.stop_after = stop_after
        self.verbose = verbose

        self.info: Dict = {}
        self.results: Dict = {}

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"  {msg}")

    # ----- Step 1: CST 拟合 -----
    def step1_fit_cst(self) -> Dict:
        self.log(f"CST 拟合: {self.dat_path.name}")
        info = fit_cst_to_dat(self.dat_path, run_xfoil=False)
        info["_dat_name"] = self.dat_path.name
        self.info = info
        _ensure_dir(self.output_dir)
        save_seed_info_json(info, self.output_dir / "seed_info.json")
        self.log(f"  AU: {np.array2string(info['au'], precision=4, separator=', ')}")
        self.log(f"  AL: {np.array2string(info['al'], precision=4, separator=', ')}")
        self.log(f"  t_max={info['t_max']:.4f}, RMSE_U={info['rmse_u']:.2e}, RMSE_L={info['rmse_l']:.2e}")
        return info

    # ----- Step 2: XFOIL 基线 -----
    def step2_xfoil_baseline(self) -> Dict:
        if self.xfoil_exe is None:
            self.log("[SKIP] 未提供 xfoil_exe，跳过 XFOIL 基线计算")
            return self.info

        self.log("XFOIL 基线计算...")
        info = self.info
        # 用拟合的 CST 重建翼型，写出临时 .dat 给 XFOIL
        xru, yru, xrl, yrl = rebuild_airfoil(
            info["au"], info["al"], info["dz_u"], info["dz_l"],
            n_points=201,
            N1=info.get("N1", 0.5), N2=info.get("N2", 1.0),
        )
        tmp_dat = self.output_dir / "_baseline_cst.dat"
        write_dat_for_xfoil(tmp_dat, f"{info['name']}_CST", xru, yru, xrl, yrl)

        xfoil_result = run_xfoil_single(
            xfoil_exe=self.xfoil_exe,
            airfoil_dat=tmp_dat,
            alpha=self.xfoil_alpha,
            Re=self.xfoil_Re,
            Mach=self.xfoil_Mach,
            Iter=self.xfoil_Iter,
        )

        info["cl"] = xfoil_result["CL"]
        info["cd"] = xfoil_result["CD"]
        info["cl_cd"] = xfoil_result["CL_CD"]
        info["xfoil_converged"] = xfoil_result["converged"]

        # 更新 seed_info.json
        save_seed_info_json(info, self.output_dir / "seed_info.json")
        # 清理临时文件
        if tmp_dat.exists():
            tmp_dat.unlink()

        self.log(f"  CL={info['cl']:.6f}, CD={info['cd']:.6f}, "
                 f"CL/CD={info['cl_cd']:.4f}, converged={info['xfoil_converged']}")
        return info

    # ----- Step 3: LHS 采样 -----
    def step3_lhs_sampling(self) -> Path:
        self.log("LHS 采样...")
        from sample_airfoils_lhs import main as sample_main

        csv_path, airfoil_dir = sample_main(
            au_base=self.info["au"],
            al_base=self.info["al"],
            dz_u=self.info["dz_u"],
            dz_l=self.info["dz_l"],
            seed_name=_safe_name(self.info["name"]),
            output_dir=str(self.output_dir),
        )

        # 复制 xfoil.exe
        if self.xfoil_exe and self.xfoil_exe.exists():
            xfoil_dst = self.output_dir / "airfoils" / "xfoil.exe"
            if not xfoil_dst.exists():
                shutil.copy2(self.xfoil_exe, xfoil_dst)

        self.results["samples_csv"] = str(csv_path)
        self.results["airfoil_dir"] = str(airfoil_dir)
        return csv_path

    # ----- Step 4: XFOIL 批量求解 -----
    def step4_xfoil_solve(self) -> Path:
        self.log("XFOIL 批量求解（可能较慢）...")
        from solve_airfoils_xfoil import main as xfoil_main

        samples_csv = self.output_dir / "samples.csv"
        out_csv = self.output_dir / "surrogate_dataset.csv"

        if not samples_csv.exists():
            raise FileNotFoundError(f"找不到 samples.csv: {samples_csv}")

        out_csv = xfoil_main(
            samples_csv=str(samples_csv),
            output_csv=str(out_csv),
            working_dir=str(self.output_dir),
        )

        self.results["surrogate_dataset"] = str(out_csv)
        return out_csv

    # ----- Step 5: 代理模型训练 -----
    def step5_surrogate(self) -> List[Dict]:
        self.log("代理模型训练...")
        from main_surrogate import main as surrogate_main

        dataset = self.output_dir / "surrogate_dataset.csv"
        if not dataset.exists():
            dataset = self.output_dir / "batch_run" / "surrogate_dataset.csv"
        if not dataset.exists():
            raise FileNotFoundError(f"找不到 surrogate_dataset.csv: {dataset}")

        model_dir = self.output_dir / "surrogate_saved_models"
        fig_dir = self.output_dir / "surrogate_plots"
        summary_xlsx = self.output_dir / "surrogate_summary.xlsx"

        all_results = surrogate_main(
            data_file=str(dataset),
            model_save_dir=str(model_dir),
            fig_save_dir=str(fig_dir),
            summary_path=str(summary_xlsx),
        )

        self.results["model_dir"] = str(model_dir)
        return all_results

    # ----- Step 6: 优化 -----
    def step6_optimize(self) -> Tuple:
        self.log(f"优化 ({self.algorithm}/{self.model_type})...")
        from main_optimization import main as opt_main

        opt_dir = self.output_dir / "optimization_results"
        info = self.info

        baseline_cl = info.get("cl", 0.5)
        if baseline_cl is None or np.isnan(baseline_cl):
            baseline_cl = 0.5
        baseline_cd = info.get("cd", 0.01)
        if baseline_cd is None or np.isnan(baseline_cd):
            baseline_cd = 0.01

        results, summary = opt_main(
            baseline_cl=baseline_cl,
            baseline_cd=baseline_cd,
            baseline_t_max=info["t_max"],
            au_base=info["au"],
            al_base=info["al"],
            dz_u=info["dz_u"],
            dz_l=info["dz_l"],
            seed_name=_safe_name(info["name"]),
            algorithm=self.algorithm,
            model_type=self.model_type,
            save_dir=str(opt_dir),
        )

        self.results["optimization_dir"] = str(opt_dir)
        return results, summary

    # ----- 主入口 -----
    def run(self) -> Dict:
        self.log(f"\n{'=' * 50}")
        self.log(f"处理: {self.dat_path.name}")
        self.log(f"{'=' * 50}")

        _ensure_dir(self.output_dir)

        stages = [
            ("sampling",  self.step3_lhs_sampling),
            ("xfoil",     self.step4_xfoil_solve),
            ("surrogate", self.step5_surrogate),
            ("full",      self.step6_optimize),
        ]

        stop_order = {
            "cst": 1, "sampling": 2, "xfoil": 3,
            "surrogate": 4, "full": 5,
        }
        max_stage = stop_order.get(self.stop_after, 5)

        # Step 1: CST fitting (always run)
        self.step1_fit_cst()

        # Step 2: XFOIL baseline (optional, fast)
        if max_stage >= 2:
            self.step2_xfoil_baseline()

        # Step 3-6: pipeline
        current_stage = 2
        for stage_name, stage_fn in stages:
            stage_num = stop_order.get(stage_name, 0)
            if stage_num <= max_stage:
                try:
                    stage_fn()
                except Exception as e:
                    self.log(f"[ERROR] {stage_name}: {e}")
                    if stage_name in ("cst", "sampling"):
                        raise  # 这些步骤必须成功
                    else:
                        self.results[f"{stage_name}_error"] = str(e)

        # 保存结果摘要
        self._save_run_summary()
        return self.results

    def _save_run_summary(self) -> None:
        summary = {
            "dat_file": str(self.dat_path),
            "output_dir": str(self.output_dir),
            "name": self.info.get("name", ""),
            "t_max": self.info.get("t_max"),
            "cl": self.info.get("cl"),
            "cd": self.info.get("cd"),
            "cl_cd": self.info.get("cl_cd"),
            "algorithm": self.algorithm,
            "model_type": self.model_type,
            **{f"ok_{k}": v for k, v in self.results.items() if not k.endswith("_error")},
        }
        with open(self.output_dir / "run_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)


# ============================================================================
# 批量运行
# ============================================================================

class BatchRunner:
    """批量对文件夹下所有 .dat 文件跑优化流水线"""

    def __init__(
        self,
        dat_dir: str | Path,
        output_root: Optional[str | Path] = None,
        xfoil_exe: Optional[str | Path] = None,
        n_samples: int = 1500,
        delta_au: float = 0.065,
        delta_al: float = 0.065,
        xfoil_alpha: float = 2.0,
        xfoil_Re: float = 6e6,
        xfoil_Mach: float = 0.3,
        algorithm: str = "RL",
        model_type: str = "kriging",
        stop_after: str = "full",
        verbose: bool = True,
    ):
        self.dat_dir = Path(dat_dir)
        self.output_root = Path(output_root) if output_root else self.dat_dir
        self.xfoil_exe = Path(xfoil_exe) if xfoil_exe else None
        self.n_samples = n_samples
        self.delta_au = delta_au
        self.delta_al = delta_al
        self.xfoil_alpha = xfoil_alpha
        self.xfoil_Re = xfoil_Re
        self.xfoil_Mach = xfoil_Mach
        self.algorithm = algorithm
        self.model_type = model_type
        self.stop_after = stop_after
        self.verbose = verbose

        self.all_results: List[Dict] = []

    def find_dat_files(self) -> List[Path]:
        """找出文件夹下所有 .dat 文件"""
        dat_files = sorted(self.dat_dir.glob("*.dat"))
        # 排除以 _ 开头的临时文件
        dat_files = [f for f in dat_files if not f.name.startswith("_")]
        return dat_files

    def run(self) -> pd.DataFrame:
        dat_files = self.find_dat_files()

        if not dat_files:
            print(f"[ERROR] 在 {self.dat_dir} 中没有找到 .dat 文件")
            return pd.DataFrame()

        print(f"找到 {len(dat_files)} 个 .dat 文件")
        print(f"输出根目录: {self.output_root}")
        print(f"停止阶段: {self.stop_after}")
        print(f"优化算法: {self.algorithm}, 代理模型: {self.model_type}")
        print()

        summary_rows = []

        for i, dat_path in enumerate(dat_files):
            name = dat_path.stem
            safe = _safe_name(name)
            output_dir = self.output_root / f"{safe}_output"

            print(f"[{i + 1}/{len(dat_files)}] {name}")

            runner = SingleAirfoilRunner(
                dat_path=dat_path,
                output_dir=output_dir,
                xfoil_exe=self.xfoil_exe,
                n_samples=self.n_samples,
                delta_au=self.delta_au,
                delta_al=self.delta_al,
                xfoil_alpha=self.xfoil_alpha,
                xfoil_Re=self.xfoil_Re,
                xfoil_Mach=self.xfoil_Mach,
                algorithm=self.algorithm,
                model_type=self.model_type,
                stop_after=self.stop_after,
                verbose=self.verbose,
            )

            try:
                results = runner.run()
                self.all_results.append(results)

                row = {
                    "index": i + 1,
                    "name": name,
                    "dat_file": str(dat_path),
                    "output_dir": str(output_dir),
                    "t_max": runner.info.get("t_max"),
                    "cl": runner.info.get("cl"),
                    "cd": runner.info.get("cd"),
                    "cl_cd": runner.info.get("cl_cd"),
                }
                # 收集优化结果
                if "optimization_dir" in results:
                    opt_summary = output_dir / "optimization_results" / "data"
                    for csv_f in opt_summary.glob("*_summary.csv"):
                        try:
                            opt_df = pd.read_csv(csv_f)
                            for _, opt_row in opt_df.iterrows():
                                row["opt_algo"] = opt_row.get("algorithm", "")
                                row["opt_cl"] = opt_row.get("opt_cl", np.nan)
                                row["opt_cd"] = opt_row.get("opt_cd", np.nan)
                                row["opt_t_max"] = opt_row.get("opt_t_max", np.nan)
                        except Exception:
                            pass

                row["status"] = "ok"
                if any(k.endswith("_error") for k in results):
                    row["status"] = "partial_error"
            except Exception as e:
                print(f"  [FAIL] {e}")
                row = {
                    "index": i + 1,
                    "name": name,
                    "dat_file": str(dat_path),
                    "output_dir": str(output_dir),
                    "status": "failed",
                    "error": str(e),
                }

            summary_rows.append(row)

        # 保存批量汇总
        df = pd.DataFrame(summary_rows)
        summary_path = self.output_root / f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"\n[OK] 批量汇总: {summary_path}")
        print(df[["name", "status", "t_max", "cl", "cl_cd"]].to_string(index=False))

        return df


# ============================================================================
# 命令行入口
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="批量翼型优化 — 对文件夹下所有 .dat 文件跑完整优化流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 完整流水线
  python run_seed_batch.py --dat_dir ./multi_seed_output/seeds/ --xfoil_exe ./4412/xfoil.exe

  # 只做 CST+LHS采样（需要快速检查）
  python run_seed_batch.py --dat_dir ./seeds/ --stop_after sampling

  # 指定优化算法
  python run_seed_batch.py --dat_dir ./seeds/ --xfoil_exe ./4412/xfoil.exe --algo GA --model nn
        """,
    )

    parser.add_argument("--dat_dir", required=True, help="包含 .dat 翼型文件的文件夹")
    parser.add_argument("-o", "--output", default=None, help="输出根目录（默认=dat_dir）")
    parser.add_argument("--xfoil_exe", default=None, help="xfoil.exe 路径")
    parser.add_argument("--n_samples", type=int, default=1500, help="LHS 采样数 (默认 1500)")
    parser.add_argument("--delta_au", type=float, default=0.065, help="AU 扰动范围")
    parser.add_argument("--delta_al", type=float, default=0.065, help="AL 扰动范围")
    parser.add_argument("--alpha", type=float, default=2.0, help="XFOIL 攻角")
    parser.add_argument("--Re", type=float, default=6e6, help="雷诺数")
    parser.add_argument("--Mach", type=float, default=0.3, help="马赫数")
    parser.add_argument("--algo", default="RL", help="优化算法: SA/GA/PSO/RL (默认 RL)")
    parser.add_argument("--model", default="kriging", help="代理模型: kriging/nn/svr (默认 kriging)")
    parser.add_argument("--stop_after", default="full",
                        choices=["cst", "sampling", "xfoil", "surrogate", "full"],
                        help="在指定阶段后停止 (默认 full)")
    parser.add_argument("-q", "--quiet", action="store_true", help="安静模式")

    args = parser.parse_args()

    runner = BatchRunner(
        dat_dir=args.dat_dir,
        output_root=args.output,
        xfoil_exe=args.xfoil_exe,
        n_samples=args.n_samples,
        delta_au=args.delta_au,
        delta_al=args.delta_al,
        xfoil_alpha=args.alpha,
        xfoil_Re=args.Re,
        xfoil_Mach=args.Mach,
        algorithm=args.algo,
        model_type=args.model,
        stop_after=args.stop_after,
        verbose=not args.quiet,
    )

    runner.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())

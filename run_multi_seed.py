# -*- coding: utf-8 -*-
"""
run_multi_seed.py — KMeans 多种子翼型优化驱动脚本
====================================================

功能：串联"KMeans 种子筛选 → CST 拟合 → LHS 采样"流程，
      为每个种子翼型生成独立的优化工作目录。

完整流程：
  Step 0:  KMeans 从翼型数据库中选出 K 个多样化的优质种子
  Step 0.5: 对每个种子 .dat 文件拟合 12 维 CST 参数
  Step 1:   对每个种子做 LHS 采样，生成样本翼型
  Step 2-3: XFOIL 求解 + 代理模型训练（需手动运行，见下方说明）
  Step 4:   对每个种子运行优化算法
  Step 5:   汇总对比所有种子的优化结果

使用方式：
  # 快速模式：只做 KMeans + CST拟合 + LHS采样
  python run_multi_seed.py --csv airfoil_db.csv --dat_dir ./dat_files/

  # 完整模式：一键跑通所有步骤（需要 xfoil.exe）
  python run_multi_seed.py --csv airfoil_db.csv --dat_dir ./dat_files/ --full

  # 指定聚类数和优化算法
  python run_multi_seed.py --csv airfoil_db.csv --dat_dir ./dat_files/ -k 5 --algo RL

依赖：本项目的 kmeans_seeds.py, fit_cst_for_seed.py, batch_run/*
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

from kmeans_seeds import KMeansSeedSelector
from fit_cst_for_seed import fit_cst_to_dat, save_seed_info_json


# ============================================================================
# 工具函数
# ============================================================================

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_name(s: str) -> str:
    """将翼型名转为安全的文件名"""
    import re
    s = s.strip()
    s = re.sub(r'[\\/:*?"<>|]', '_', s)
    s = re.sub(r'\s+', '_', s)
    return s[:80]


# ============================================================================
# 主驱动类
# ============================================================================

class MultiSeedRunner:
    """
    多种子翼型优化驱动。

    流程：
        1. KMeans 从数据库选种子
        2. 对每个种子拟合 CST
        3. 对每个种子做 LHS 采样
        4. (可选) 运行完整优化流水线
    """

    def __init__(
        self,
        csv_path: str | Path,
        dat_dir: Optional[str | Path] = None,
        output_root: str | Path = "./multi_seed_output",
        aoa: Optional[float] = None,
        n_clusters: int = 3,
        excellent_percentile: float = 0.70,
        random_state: int = 42,
        # LHS 采样参数
        n_samples: int = 1500,
        delta_au: float = 0.065,
        delta_al: float = 0.065,
        # 优化参数
        algorithm: str = "RL",
        model_type: str = "kriging",
        # 控制开关
        full_pipeline: bool = False,
        skip_sampling: bool = False,
        verbose: bool = True,
    ):
        self.csv_path = Path(csv_path)
        self.dat_dir = Path(dat_dir) if dat_dir else None
        self.output_root = Path(output_root)
        self.aoa = aoa
        self.n_clusters = n_clusters
        self.excellent_percentile = excellent_percentile
        self.random_state = random_state
        self.n_samples = n_samples
        self.delta_au = delta_au
        self.delta_al = delta_al
        self.algorithm = algorithm
        self.model_type = model_type
        self.full_pipeline = full_pipeline
        self.skip_sampling = skip_sampling
        self.verbose = verbose

        # 结果存储
        self.seeds_df: Optional[pd.DataFrame] = None
        self.seed_infos: List[Dict] = []
        self.seed_results: List[Dict] = []

    # ----- Step 0: KMeans 选种子 -----
    def step0_select_seeds(self) -> pd.DataFrame:
        """运行 KMeans 聚类，选出多样化种子翼型"""
        if self.verbose:
            print("=" * 60)
            print("Step 0: KMeans 种子筛选")
            print("=" * 60)

        selector = KMeansSeedSelector(
            csv_path=self.csv_path,
            dat_dir=self.dat_dir,
            output_dir=str(self.output_root),
            aoa=self.aoa,
            n_clusters=self.n_clusters,
            excellent_percentile=self.excellent_percentile,
            random_state=self.random_state,
            export_dat=True,
            verbose=self.verbose,
        )
        seeds_df, summary_df = selector.run()
        self.seeds_df = seeds_df

        if self.verbose:
            print(f"\n选出 {len(seeds_df)} 个种子翼型")

        return seeds_df

    # ----- Step 0.5: CST 拟合 -----
    def step0_5_fit_cst(self) -> List[Dict]:
        """对每个种子翼型拟合 12 维 CST 参数"""
        if self.verbose:
            print("\n" + "=" * 60)
            print("Step 0.5: CST 参数拟合")
            print("=" * 60)

        if self.seeds_df is None:
            raise RuntimeError("请先运行 step0_select_seeds()")

        seeds_dat_dir = self.output_root / "seeds"
        seed_infos = []

        for i, (_, seed) in enumerate(self.seeds_df.iterrows()):
            cid = int(seed.get("cluster_id", i + 1))
            fn = seed.get("Filename", f"seed_{cid}")

            # 找种子 .dat 文件
            safe = _safe_name(str(fn))
            dat_candidates = list(seeds_dat_dir.glob(f"*cluster{cid}*{safe}*.dat"))
            if not dat_candidates:
                dat_candidates = list(seeds_dat_dir.glob(f"*cluster{cid}*.dat"))
            if not dat_candidates:
                print(f"  [WARN] 找不到 Cluster {cid} 的 .dat 文件: {fn}")
                continue

            seed_dat = dat_candidates[0]

            if self.verbose:
                print(f"\n  Cluster {cid}: {fn}")
                print(f"    .dat: {seed_dat}")

            # 拟合 CST
            info = fit_cst_to_dat(seed_dat, run_xfoil=False)
            info["cluster_id"] = cid
            info["original_filename"] = str(fn)
            info["seed_dat"] = str(seed_dat)
            # 用数据库中的原始文件名覆盖（.dat 文件可能无名称头导致 stem 被误用）
            info["name"] = str(fn)

            # 保存种子信息 JSON
            seed_dir = self.output_root / f"seed_{cid}_{_safe_name(str(fn))}"
            _ensure_dir(seed_dir)
            json_path = seed_dir / "seed_info.json"
            save_seed_info_json(info, json_path)

            if self.verbose:
                print(f"    AU: {np.array2string(info['au'], precision=4, separator=', ')}")
                print(f"    AL: {np.array2string(info['al'], precision=4, separator=', ')}")
                print(f"    DZ_U={info['dz_u']:.6f}, DZ_L={info['dz_l']:.6f}")
                print(f"    t_max={info['t_max']:.4f}, RMSE_U={info['rmse_u']:.2e}, RMSE_L={info['rmse_l']:.2e}")
                print(f"    [OK] seed_info.json -> {json_path}")

            seed_infos.append(info)

        self.seed_infos = seed_infos
        return seed_infos

    # ----- Step 1: LHS 采样 -----
    def step1_lhs_sampling(self) -> List[Dict]:
        """对每个种子做 LHS 采样"""
        if self.verbose:
            print("\n" + "=" * 60)
            print("Step 1: LHS 采样")
            print("=" * 60)

        if not self.seed_infos:
            raise RuntimeError("请先运行 step0_5_fit_cst()")

        # 动态导入采样模块
        sys.path.insert(0, str(PROJECT_ROOT / "batch_run"))
        from sample_airfoils_lhs import main as sample_main

        results = []
        for info in self.seed_infos:
            cid = info["cluster_id"]
            name = info["name"]
            seed_dir = Path(info.get("_seed_dir",
                           self.output_root / f"seed_{cid}_{_safe_name(str(name))}"))
            info["_seed_dir"] = str(seed_dir)

            if self.verbose:
                print(f"\n  Cluster {cid}: {name}")
                print(f"    采样目录: {seed_dir}")

            # 运行 LHS 采样
            csv_path, airfoil_dir = sample_main(
                au_base=info["au"],
                al_base=info["al"],
                dz_u=info["dz_u"],
                dz_l=info["dz_l"],
                seed_name=f"seed_{cid}",
                output_dir=str(seed_dir),
            )

            # 复制 xfoil.exe 到采样目录（方便后续 XFOIL 求解）
            xfoil_src = PROJECT_ROOT / "4412" / "xfoil.exe"
            if not xfoil_src.exists():
                xfoil_src = PROJECT_ROOT / "batch_run" / "airfoils" / "xfoil.exe"
            if xfoil_src.exists():
                dst = seed_dir / "airfoils" / "xfoil.exe"
                if not dst.exists():
                    shutil.copy2(xfoil_src, dst)

            results.append({
                "cluster_id": cid,
                "name": name,
                "seed_dir": str(seed_dir),
                "samples_csv": str(csv_path),
                "airfoil_dir": str(airfoil_dir),
                "cst_info": info,
            })

        self.seed_results = results
        return results

    # ----- 完整流水线 -----
    def run_full_pipeline_for_seed(self, seed_result: Dict) -> Dict:
        """对单个种子运行完整流水线（XFOIL + 代理模型 + 优化）"""
        seed_dir = Path(seed_result["seed_dir"])
        cst_info = seed_result["cst_info"]

        # Step 2: XFOIL 求解
        if self.verbose:
            print(f"\n  >>> XFOIL 求解: {seed_dir}")

        sys.path.insert(0, str(PROJECT_ROOT / "batch_run"))
        # 临时切换工作目录到种子目录
        import os as _os
        _orig_cwd = _os.getcwd()

        try:
            _os.chdir(str(seed_dir))
            from solve_airfoils_xfoil import main as xfoil_main
            xfoil_main()
        finally:
            _os.chdir(_orig_cwd)

        # Step 3: 代理模型训练
        surrogate_dataset = seed_dir / "surrogate_dataset.csv"
        if not surrogate_dataset.exists():
            surrogate_dataset = seed_dir / "batch_run" / "surrogate_dataset.csv"

        if surrogate_dataset.exists():
            if self.verbose:
                print(f"\n  >>> 代理模型训练: {seed_dir}")

            # 把 surrogate_dataset.csv 复制到正确位置
            batch_dir = seed_dir / "batch_run"
            _ensure_dir(batch_dir)
            target_csv = batch_dir / "surrogate_dataset.csv"
            if not target_csv.exists():
                shutil.copy2(surrogate_dataset, target_csv)

            # 训练代理模型 - 需要修改路径
            # 这里比较复杂，因为 main_surrogate.py 有硬编码的路径
            # 暂时跳过，让用户手动运行
            if self.verbose:
                print(f"    [INFO] 代理模型训练需要手动运行:")
                print(f"    cd {seed_dir}")
                print(f"    python {PROJECT_ROOT / 'surrogate' / 'main_surrogate.py'}")

        # Step 4: 优化
        if self.verbose:
            print(f"\n  >>> 优化: {seed_dir}")

        sys.path.insert(0, str(PROJECT_ROOT / "youhua"))
        from main_optimization import main as opt_main

        opt_results, opt_summary = opt_main(
            baseline_cl=cst_info.get("cl") or 0.5,
            baseline_cd=cst_info.get("cd") or 0.01,
            baseline_t_max=cst_info["t_max"],
            au_base=cst_info["au"],
            al_base=cst_info["al"],
            dz_u=cst_info["dz_u"],
            dz_l=cst_info["dz_l"],
            seed_name=f"seed_{seed_result['cluster_id']}",
            algorithm=self.algorithm,
            model_type=self.model_type,
            save_dir=str(seed_dir / "optimization_results"),
        )

        return {
            **seed_result,
            "optimization_results": opt_results,
            "optimization_summary": opt_summary,
        }

    # ----- 主入口 -----
    def run(self) -> Dict:
        """执行完整的多种子筛选流程"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        _ensure_dir(self.output_root)

        # Step 0: KMeans 选种子
        seeds_df = self.step0_select_seeds()

        # Step 0.5: CST 拟合
        seed_infos = self.step0_5_fit_cst()

        if not seed_infos:
            raise RuntimeError("没有成功拟合任何种子的 CST 参数")

        # Step 1: LHS 采样
        if not self.skip_sampling:
            seed_results = self.step1_lhs_sampling()
        else:
            seed_results = [
                {
                    "cluster_id": info["cluster_id"],
                    "name": info["name"],
                    "seed_dir": info.get("_seed_dir", ""),
                    "cst_info": info,
                }
                for info in seed_infos
            ]
            self.seed_results = seed_results

        # 保存多种子汇总
        summary = self._save_multi_seed_summary(timestamp)

        # (可选) 完整流水线
        if self.full_pipeline:
            if self.verbose:
                print("\n" + "=" * 60)
                print("完整流水线模式 (XFOIL + 代理模型 + 优化)")
                print("=" * 60)

            final_results = []
            for sr in seed_results:
                try:
                    final = self.run_full_pipeline_for_seed(sr)
                    final_results.append(final)
                except Exception as e:
                    print(f"  [ERROR] 种子 {sr['cluster_id']} ({sr['name']}) 失败: {e}")
                    final_results.append({**sr, "error": str(e)})

            self.seed_results = final_results
            self._save_final_summary(timestamp)

        # 打印后续步骤
        self._print_next_steps()

        return summary

    def _save_multi_seed_summary(self, timestamp: str) -> Dict:
        """保存多种子信息汇总"""
        summary = {
            "timestamp": timestamp,
            "n_clusters": self.n_clusters,
            "excellent_percentile": self.excellent_percentile,
            "csv_source": str(self.csv_path),
            "seeds": [],
        }

        for info in self.seed_infos:
            seed_entry = {
                "cluster_id": info["cluster_id"],
                "name": info["name"],
                "original_filename": info.get("original_filename", ""),
                "t_max": info["t_max"],
                "dz_u": info["dz_u"],
                "dz_l": info["dz_l"],
                "au": info["au"].tolist(),
                "al": info["al"].tolist(),
                "rmse_u": info["rmse_u"],
                "rmse_l": info["rmse_l"],
                "seed_dir": info.get("_seed_dir", ""),
                "seed_dat": info.get("seed_dat", ""),
            }
            summary["seeds"].append(seed_entry)

        # 保存 JSON
        summary_path = self.output_root / "multi_seed_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        if self.verbose:
            print(f"\n[OK] 多种子汇总: {summary_path}")

        return summary

    def _save_final_summary(self, timestamp: str) -> None:
        """保存最终优化结果汇总"""
        rows = []
        for sr in self.seed_results:
            row = {
                "cluster_id": sr["cluster_id"],
                "name": sr["name"],
                "seed_dir": sr.get("seed_dir", ""),
            }
            if "optimization_summary" in sr:
                for _, opt_row in sr["optimization_summary"].iterrows():
                    row["algorithm"] = opt_row.get("algorithm", "")
                    row["baseline_cl"] = opt_row.get("baseline_cl", np.nan)
                    row["baseline_cd"] = opt_row.get("baseline_cd", np.nan)
                    row["opt_cl"] = opt_row.get("opt_cl", np.nan)
                    row["opt_cd"] = opt_row.get("opt_cd", np.nan)
                    row["opt_t_max"] = opt_row.get("opt_t_max", np.nan)
                    rows.append(row.copy())
            elif "error" in sr:
                row["error"] = sr["error"]
                rows.append(row)

        if rows:
            df = pd.DataFrame(rows)
            csv_path = self.output_root / "final_all_seeds_results.csv"
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            if self.verbose:
                print(f"[OK] 最终结果: {csv_path}")

    def _print_next_steps(self) -> None:
        """打印后续手动步骤说明"""
        print("\n" + "=" * 60)
        print("后续步骤")
        print("=" * 60)

        for sr in self.seed_results:
            cid = sr["cluster_id"]
            name = sr.get("name", "")
            # 如果 name 是坐标值或空字符串，尝试用 original_filename
            if not name or name.startswith(" ") or "e+" in str(name).lower():
                cst_info = sr.get("cst_info", {})
                name = cst_info.get("original_filename", f"seed_{cid}")
            seed_dir = sr.get("seed_dir", "")

            print(f"\n--- Cluster {cid}: {name} ---")
            if seed_dir:
                print(f"  工作目录: {seed_dir}")
            print(f"  seed_info.json 已保存（含 12 维 CST 参数 + t_max）")
            print(f"  1. XFOIL 求解:")
            print(f"     cd {seed_dir}")
            print(f"     python {PROJECT_ROOT / 'batch_run' / 'solve_airfoils_xfoil.py'}")
            print(f"  2. 代理模型训练:")
            print(f"     cd {PROJECT_ROOT / 'surrogate'}")
            print(f"     (需要先把 {seed_dir}/surrogate_dataset.csv")
            print(f"      复制到 batch_run/ 并修改 main_surrogate.py 中的路径)")
            print(f"  3. 优化:")
            print(f"     python {PROJECT_ROOT / 'youhua' / 'main_optimization.py'}")

        print(f"\n或者直接使用参数化接口:")
        print(f"  from youhua.main_optimization import main as opt_main")
        print(f"  opt_main(au_base=seed_au, al_base=seed_al, ...)")
        print(f"\n所有种子信息已保存在: {self.output_root}")


# ============================================================================
# 命令行入口
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="KMeans 多种子翼型优化驱动",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 快速模式：KMeans + CST拟合 + LHS采样
  python run_multi_seed.py --csv airfoil_db.csv --dat_dir ./dat_files/

  # 选5个种子
  python run_multi_seed.py --csv airfoil_db.csv --dat_dir ./dat_files/ -k 5

  # 只做KMeans筛选和CST拟合，不做采样
  python run_multi_seed.py --csv airfoil_db.csv --dat_dir ./dat_files/ --skip_sampling
        """,
    )

    parser.add_argument("--csv", required=True, help="翼型数据库 CSV 路径")
    parser.add_argument("--dat_dir", default=None, help=".dat 翼型坐标文件夹")
    parser.add_argument("-o", "--output", default="./multi_seed_output", help="输出根目录")
    parser.add_argument("--aoa", type=float, default=None, help="攻角筛选")
    parser.add_argument("-k", "--n_clusters", type=int, default=3, help="聚类簇数")
    parser.add_argument("--percentile", type=float, default=0.70, help="优秀 L/D 分位数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--n_samples", type=int, default=1500, help="LHS 采样数")
    parser.add_argument("--delta_au", type=float, default=0.065, help="AU 扰动范围")
    parser.add_argument("--delta_al", type=float, default=0.065, help="AL 扰动范围")
    parser.add_argument("--algo", default="RL", help="优化算法 (SA/GA/PSO/RL/ALL)")
    parser.add_argument("--model", default="kriging", help="代理模型 (kriging/nn/svr)")
    parser.add_argument("--full", action="store_true", help="运行完整流水线")
    parser.add_argument("--skip_sampling", action="store_true", help="跳过 LHS 采样")
    parser.add_argument("-q", "--quiet", action="store_true", help="安静模式")

    args = parser.parse_args()

    runner = MultiSeedRunner(
        csv_path=args.csv,
        dat_dir=args.dat_dir,
        output_root=args.output,
        aoa=args.aoa,
        n_clusters=args.n_clusters,
        excellent_percentile=args.percentile,
        random_state=args.seed,
        n_samples=args.n_samples,
        delta_au=args.delta_au,
        delta_al=args.delta_al,
        algorithm=args.algo,
        model_type=args.model,
        full_pipeline=args.full,
        skip_sampling=args.skip_sampling,
        verbose=not args.quiet,
    )

    summary = runner.run()

    print(f"\n[OK] 多种子筛选完成，输出目录: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

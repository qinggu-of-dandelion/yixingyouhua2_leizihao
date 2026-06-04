# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## 项目概述

基于 KMeans 聚类 + 代理模型 + 元启发式算法的翼型气动优化系统。核心思路：从翼型数据库中选出多样化优质种子，然后对每个种子独立做局部优化（LHS 采样 → XFOIL 求解 → 代理模型训练 → 优化搜索），最终选最优翼型。

## 环境

```bash
conda activate airfoil_opt
```

Python: `D:\Anacoda\envs\airfoil_opt\python.exe`

## 快速开始（一键运行）

```bash
# Step 1: KMeans 从数据库选种子（k=6 个多样化优质翼型）
python run_multi_seed.py --csv ../airfoil_opt/data/final_results.csv --dat_dir ../airfoil_opt/data/processed -k 6 --aoa 4.0

# Step 2: 对每个种子跑完整优化流水线
python run_seed_batch.py --dat_dir ./multi_seed_output/seeds/ --xfoil_exe ./4412/xfoil.exe --algo GA --model kriging
```

输出在 `multi_seed_output/seeds/<翼型名>_output/optimization_results/data/`，XFOIL 验证 JSON 为 `xfoil_verify.json`。

## 目录结构

```
yixingyouhua2_leizihao/
├── kmeans_seeds.py              # KMeans 种子筛选（独立模块，零项目依赖）
├── fit_cst_for_seed.py          # 对任意 .dat 拟合 12 维 CST 参数
├── run_multi_seed.py            # 多种子驱动：KMeans → CST拟合 → 种子 .dat 输出
├── run_seed_batch.py            # 批量优化：对文件夹下所有 .dat 逐一跑全流程
│
├── batch_run/
│   ├── config.py                # 全局配置（CST参数、LHS范围、XFOIL工况）
│   ├── sample_airfoils_lhs.py   # LHS 采样（main() 已参数化，支持外部基线）
│   └── solve_airfoils_xfoil.py  # XFOIL 批量求解（main() 已参数化）
│
├── surrogate/
│   ├── main_surrogate.py        # 代理模型训练（main() 已参数化）
│   ├── kriging_model.py         # Kriging（高斯过程回归）
│   ├── svr_model.py             # SVR（支持向量回归）
│   ├── nn_model.py              # NN（BP 神经网络 MLPRegressor）
│   └── sample_size_study.py     # 样本数-精度影响研究（可选）
│
├── youhua/
│   ├── main_optimization.py     # 优化主程序（main() 已参数化）
│   ├── xfoil_verify_results.py  # XFOIL 真实验证（main() 已参数化）
│   ├── simulated_annealing.py   # 模拟退火 (SA)
│   ├── genetic_algorithm.py     # 遗传算法 (GA)
│   ├── particle_swarm.py        # 粒子群 (PSO)
│   └── reinforcement_learning.py # 强化学习 DDPG (RL, 需要 PyTorch)
│
├── 4412/
│   ├── NACA4412.dat             # 基准翼型（用于测试对比）
│   ├── xfoil.exe                # XFOIL 6.99
│   └── cst_xfoil.py             # 原始 CST 拟合工具（独立使用）
│
└── multi_seed_output/           # 输出目录（gitignore）
    └── seeds/
        ├── seed_cluster*_*.dat  # 种子翼型
        └── *_output/            # 每个种子的独立优化结果
```

## 数据流

```
airfoil_opt 数据库 (CSV + .dat)
    │
    ▼
kmeans_seeds.py          → 从 ~2900 翼型中聚类选 k 个多样优质种子
    │                      输出: multi_seed_output/seeds/*.dat
    ▼
fit_cst_for_seed.py      → 对每个种子 .dat 拟合 12 维 CST 参数
    │                      输出: seed_info.json
    ▼
sample_airfoils_lhs.py   → 在种子 CST 邻域做 LHS 采样
    │                      输出: samples.csv + airfoils/*.dat
    ▼
solve_airfoils_xfoil.py  → XFOIL 批量求解气动性能
    │                      输出: surrogate_dataset.csv
    ▼
main_surrogate.py        → 训练 3 种代理模型 × 3 个目标 = 9 个模型
    │                      输出: surrogate_saved_models/*.pkl
    ▼
main_optimization.py     → GA/SA/PSO 搜索最优 CST 参数
    │                      输出: optimization_results/
    ▼
xfoil_verify_results.py  → XFOIL 真实验证优化翼型
                           输出: xfoil_verify.json
```

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-k` / `n_clusters` | 3 | KMeans 聚类数 |
| `--aoa` | 4.0 | 攻角筛选 |
| `--n_samples` | 1500 | LHS 采样数 |
| `--algo` | RL | 优化算法: SA/GA/PSO/RL |
| `--model` | kriging | 代理模型: kriging/svr/nn |
| `--xfoil_exe` | — | xfoil.exe 路径（必须提供） |
| `--stop_after` | full | 流水线停止阶段: cst/sampling/xfoil/surrogate/full |

## 控制阶段（run_seed_batch.py）

```bash
# 只做 CST + LHS 采样（快速检查）
python run_seed_batch.py --dat_dir ./seeds/ --stop_after sampling

# 做到 XFOIL 求解（得 surrogate_dataset.csv）
python run_seed_batch.py --dat_dir ./seeds/ --xfoil_exe ./4412/xfoil.exe --stop_after xfoil

# 完整流水线
python run_seed_batch.py --dat_dir ./seeds/ --xfoil_exe ./4412/xfoil.exe --algo GA --stop_after full
```

## 所有 main() 均已参数化

底层脚本 `main()` 支持被外部调用时传入自定义参数，独立运行时使用硬编码默认值：

- `sample_airfoils_lhs.main(au_base, al_base, dz_u, dz_l, seed_name, output_dir)`
- `solve_airfoils_xfoil.main(samples_csv, output_csv, working_dir)`
- `main_surrogate.main(data_file, model_save_dir, fig_save_dir, summary_path)`
- `main_optimization.main(baseline_cl, baseline_cd, baseline_t_max, au_base, al_base, model_dir, save_dir, algorithm, model_type)`
- `xfoil_verify_results.main(opt_result_dir, case_name, model_type, xfoil_exe, baseline_dat)`

## 优化目标

最小化 Cd，约束条件 Cl ≥ baseline_Cl、t_max ≥ baseline_tmax。罚函数系数 1e7。

## 注意事项

- **XFOIL 串行**：Windows spawn 多进程无法传递覆盖后的全局变量，自定义工作目录时强制串行执行
- **RL 需 PyTorch**：如未安装，将 `--algo` 设为 GA/SA/PSO
- **goe529 种子的代理模型可能预测负 Cd**：该种子区域数据稀疏，kriging 外推不可靠
- **文件名不宜过长**：XFOIL 对过长的 .dat 文件名支持不好，已通过短名复制规避
- **CST 维度不同不冲突**：airfoil_opt 数据库用 8 维 CST，本项目用 12 维，通过 .dat 文件传递松耦合

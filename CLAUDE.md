# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

NACA 4412 翼型气动优化项目（leizihao 优化版），在原始版本基础上对代理模型训练、优化算法参数、以及新增强化学习优化方法进行了全面改进。使用代理模型 + 元启发式算法 + 强化学习进行翼型外形优化。

## Conda 环境

所有脚本都在 `airfoil_opt` 环境中运行：

```bash
conda activate airfoil_opt
```

或直接使用 Python 路径：`D:\Anacoda\envs\airfoil_opt\python.exe`

## 流水线执行顺序（必须按序执行）

```
1. batch_run/sample_airfoils_lhs.py    → 生成 samples.csv + airfoils/*.dat
2. batch_run/solve_airfoils_xfoil.py   → 生成 surrogate_dataset.csv
3. surrogate/sample_size_study.py      → （可选）样本数-精度影响研究
4. surrogate/main_surrogate.py         → 训练模型，输出 surrogate_saved_models/*.pkl + *_meta.json
5. youhua/main_optimization.py         → 加载模型，运行优化（可选 SA/GA/PSO/RL）
6. youhua/xfoil_verify_results.py      → XFOIL 验证优化结果
```

每步依赖上一步的输出，不能跳过。第 3 步为可选的分析步骤。

## 目录结构

| 目录 | 用途 |
|------|------|
| `4412/` | 基准翼型（NACA4412.dat）、CST参数拟合、XFOIL 可执行文件 |
| `batch_run/` | 设计空间采样 + XFOIL 批量求解 |
| `surrogate/` | 代理模型训练（Kriging/SVR/NN）+ 样本数影响研究 |
| `youhua/` | 优化算法（SA/GA/PSO/RL）+ XFOIL 验证 |

## 相比原始版本的核心改进

### 一、代理模型训练优化（surrogate/）

**1. 并行训练**
- 使用 `ProcessPoolExecutor` 多进程并行训练不同模型-目标组合
- 配置项：`PARALLEL = True`，`MAX_WORKERS = 4`
- 通过环境变量 `SURROGATE_MAX_TASKS` 可限制调试时的任务数

**2. NN 分目标差异化配置（NN_CONFIG_BY_TARGET）**
- cl：较小网络 (128,64)，alpha=1e-4，lr=1e-3，epochs=600
- cd：增强网络 (256,128)，alpha=3e-2（强正则），lr=5e-4，epochs=800
- t_max：增强网络 (256,128)，alpha=1e-4，lr=1e-3，epochs=700
- 原版三种目标共用同一套 NN 超参数 (128,64)，epochs=500

**3. 目标变换（TARGET_TRANSFORM）**
- NN 训练 cd 时使用 `log(cd)` 变换，控制相对误差
- 保存模型时同时保存 `*_meta.json`，记录 `target_transform`
- 优化程序 Evaluator 自动读取 meta.json 并做 `exp()` 还原
- 原版直接在原始尺度上训练

**4. SVR 超参数增强**
- C: 10.0 → 30.0（更强拟合能力）
- epsilon: 0.01 → 0.001（更精细的回归精度）

**5. 样本数影响研究（新增 sample_size_study.py）**
- 系统比较不同训练集大小 [100, 200, 400, 600, 800, "all"] 对精度的影响
- 多随机种子重复实验，输出 RMSE/R²/MAPE 随样本数变化曲线
- 也使用多进程并行加速

### 二、优化算法参数优化（youhua/）

**1. 模拟退火（SA）增强**
- max_iter: 2000 → 6000
- cooling_rate: 0.985 → 0.997（更慢降温，搜索更充分）
- step_ratio: 0.05 → 0.04（更精细的邻域搜索）

**2. 遗传算法（GA）增强**
- pop_size: 60 → 150（更大种群）
- generations: 150 → 500（更多迭代代数）
- elite_size: 2 → 4（更强精英保留）
- mutation_rate: 0.15 → 0.2

**3. 粒子群算法（PSO）增强**
- swarm_size: 60 → 120（更大粒子群）
- max_iter: 200 → 500（更多迭代）
- w: 0.85 → 0.72（更快收敛）
- c1/c2: 1.3 → 1.7（更强的个体/社会学习）
- v_ratio: 0.25 → 0.10（更精细的搜索步长）

**4. 罚函数系数增强**
- PENALTY_CL: 1e4 → 1e7
- PENALTY_TMAX: 1e4 → 1e7
- 更强的约束违反惩罚，确保可行解

### 三、强化学习优化方法（新增 youhua/reinforcement_learning.py）

基于 DDPG（Deep Deterministic Policy Gradient）的连续动作空间强化学习优化器。

**核心架构：**
- Actor-Critic 结构，使用 PyTorch 实现
- Actor：输入状态 → 输出动作（12维 CST 参数增量）
- Critic：输入状态+动作 → 输出 Q 值
- 隐藏层 (128,128) 配合 LayerNorm 归一化
- 目标网络软更新（tau=0.01）
- 经验回放缓冲区（capacity=50000）

**状态表示（17维）：**
- 12 维归一化 CST 参数（映射到 [-1,1]）
- 5 维额外信息：cl、cd×1000、t_max×10、log1p(penalty)、feasible标志

**奖励设计：**
- 基于相对改善比例（而非绝对差值，避免数值淹没）
- 可行性奖励：feasible → +0.1，不可行 → -0.1
- 全局最优改善额外奖励

**探索策略：**
- Warmup 随机探索阶段（800步）
- 高斯噪声叠加，随 episode 衰减（noise_decay=0.9985）
- 自适应探索重启：连续 patience(25) 轮无改善时放大噪声并随机重启
- 步长随 episode 线性衰减

**RL 超参数（RL_PARAMS）：**
- episodes=300, steps_per_episode=35
- step_ratio=0.06, gamma=0.98
- actor_lr=1e-4, critic_lr=3e-4
- batch_size=256, updates_per_step=2
- warmup_steps=800, exploration_noise=0.20
- patience=25（自适应重启阈值）

### 四、采样配置优化（batch_run/config.py）

- N_SAMPLES: 1000 → 1500（增加样本量）
- DELTA_AU/AL: 0.075 → 0.065（收缩扰动范围，聚焦局部优化）
- BASELINE 参数从 Excel 自动读取（`read_baseline_from_xlsx()`），换翼型只需改 Excel 路径
- 原版 BASELINE 参数硬编码在 config.py 中

### 五、Evaluator 增强（youhua/main_optimization.py）

- 自动读取代理模型的 `*_meta.json`，识别目标变换类型
- 新增 `inverse_target_transform()` 方法，在预测时自动还原变换
- 模型文件字典增加 `meta` 路径字段

## 关键架构决策

- **12 个设计变量**：6 个上表面 CST 系数 (Au_0..Au_5) + 6 个下表面 CST 系数 (Al_0..Al_5)，dz 项固定
- **1500 个 LHS 样本**（原版 1000），扰动范围 baseline ± 0.065（原版 0.075）
- **XFOIL 工况**：alpha=2°, Re=6e6, Mach=0.3
- **三种代理模型**分别对 cl、cd、t_max 三个目标独立训练
- **NN 分目标配置**：cl/cd/t_max 使用不同的网络结构和超参数
- **cd 目标变换**：NN 训练 cd 时使用 log(cd) 训练，预测时 exp 还原
- **优化目标**：最小化 cd，约束 cl >= baseline_cl，t_max >= baseline_tmax
- **优化器搜索空间**相对采样范围收缩 5%（避免外推），约束通过罚函数（系数 1e7）处理
- **默认使用 Kriging 模型 + RL 算法**进行优化
- **CST 翼型点密度**：N_POINTS = 301（原版 201）

## 运行模式

**代理模型训练（main_surrogate.py）：**
- `RUN_MODE = "all"`：一次性训练所有 3×3=9 个模型（默认并行）
- `RUN_MODE = "single"`：单独训练一个模型+目标
- `PARALLEL = True` + `MAX_WORKERS = 4`：多进程并行

**优化算法（main_optimization.py）：**
- `ALGORITHM = "RL"`：强化学习（默认）
- `ALGORITHM = "SA"`：模拟退火
- `ALGORITHM = "GA"`：遗传算法
- `ALGORITHM = "PSO"`：粒子群算法
- `ALGORITHM = "ALL"`：全部对比
- `MODEL_TYPE = "kriging"`：使用的代理模型类型（kriging/svr/nn）

## 并行训练注意

- `surrogate/main_surrogate.py` 和 `surrogate/sample_size_study.py` 使用 `ProcessPoolExecutor`
- 必须在 `if __name__ == "__main__"` 中调用 `multiprocessing.freeze_support()`（已添加）
- 环境变量限制调试：`$env:SURROGATE_MAX_TASKS='2'` 或 `$env:SAMPLE_STUDY_MAX_TASKS='6'`
- 并行模式仅 CPU，不涉及 GPU

## 常见问题

- `ModuleNotFoundError: No module named 'sklearn'` → 确认 conda 环境是 `airfoil_opt` 而不是 base
- `ModuleNotFoundError: No module named 'torch'` → RL 算法需要 PyTorch；如不使用 RL 则不影响
- `FileNotFoundError: surrogate_dataset.csv` → 还没跑 `batch_run/solve_airfoils_xfoil.py`
- `FileNotFoundError: kriging_cl.pkl` → 还没跑 `surrogate/main_surrogate.py`
- `FileNotFoundError: kriging_cl_meta.json` → 旧版模型没有 meta 文件，Evaluator 会自动降级为 identity 变换
- 优化结果不满足约束 → 调大 `PENALTY_CL` 和 `PENALTY_TMAX`（当前均为 1e7）
- RL 训练不收敛 → 增加 `warmup_steps`、调大 `exploration_noise` 或减少 `noise_decay`

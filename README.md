# 基于 KMeans 聚类的翼型气动优化系统

从翼型数据库中自动筛选多样化优质种子，对每个种子独立进行局部代理模型优化，实现减阻设计。

## 环境要求

### Python 库

```
numpy
pandas
scipy
scikit-learn>=1.6
matplotlib
pyyaml
joblib
openpyxl
```

### 可选（RL 算法需要）

```
pytorch
```

### 外部工具

- **XFOIL 6.99**（`4412/xfoil.exe`）：用于翼型气动性能求解

### Conda 环境

```bash
conda activate airfoil_opt
```

## 快速开始

两步完成从数据库到优化结果的全流程：

```bash
# 1. 从翼型数据库选 6 个多样化优质种子
python run_multi_seed.py \
    --csv ../airfoil_opt/data/final_results.csv \
    --dat_dir ../airfoil_opt/data/processed \
    -k 6 --aoa 4.0

# 2. 对每个种子跑完整优化流水线（需要 xfoil.exe）
python run_seed_batch.py \
    --dat_dir ./multi_seed_output/seeds/ \
    --xfoil_exe ./4412/xfoil.exe \
    --algo GA --model kriging
```

结果在每个种子的 `_output/optimization_results/data/` 目录下。

## 分步运行

如果想逐步控制流程：

```bash
# Step 1: 只做 KMeans 筛选 + CST 拟合（不采样）
python run_multi_seed.py --csv <数据库.csv> --dat_dir <.dat文件夹> -k 6 --skip_sampling

# Step 2: 只做 CST 拟合 + LHS 采样（不 XFOIL）
python run_seed_batch.py --dat_dir ./seeds/ --stop_after sampling

# Step 3: 做到 XFOIL 求解完成
python run_seed_batch.py --dat_dir ./seeds/ --xfoil_exe ./4412/xfoil.exe --stop_after xfoil

# Step 4: 做到代理模型训练完成
python run_seed_batch.py --dat_dir ./seeds/ --xfoil_exe ./4412/xfoil.exe --stop_after surrogate

# Step 5: 完整流水线（含优化）
python run_seed_batch.py --dat_dir ./seeds/ --xfoil_exe ./4412/xfoil.exe --stop_after full
```

## 参数说明

### run_multi_seed.py（KMeans 种子筛选）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--csv` | **必填** | 翼型数据库 CSV 路径（需含 CST 系数、Cl、Cd 列） |
| `--dat_dir` | 无 | .dat 翼型坐标文件夹（用于提取几何特征 t_max、camber_max） |
| `-k, --n_clusters` | 3 | KMeans 聚类簇数，即选出几个种子翼型 |
| `--aoa` | 无 | 攻角筛选（如 4.0），不指定则不过滤 |
| `--percentile` | 0.70 | "优秀"翼型的 L/D 分位数阈值（0.70 = 前 30%） |
| `--seed` | 42 | 随机种子（影响聚类结果） |
| `--skip_sampling` | False | 跳过 LHS 采样步骤，只做筛选和 CST 拟合 |
| `-o, --output` | ./multi_seed_output | 输出根目录 |
| `-q, --quiet` | False | 安静模式，减少打印 |

### run_seed_batch.py（批量优化流水线）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dat_dir` | **必填** | 包含 .dat 翼型文件的文件夹 |
| `--xfoil_exe` | 无 | xfoil.exe 路径（做 XFOIL 求解时必须提供） |
| `--n_samples` | 1500 | 每个种子的 LHS 采样数（越多代理模型越准，越慢） |
| `--delta_au` | 0.065 | 上表面 CST 系数扰动范围（越大搜索越广，代理模型越不准） |
| `--delta_al` | 0.065 | 下表面 CST 系数扰动范围 |
| `--alpha` | 2.0 | XFOIL 计算攻角（度） |
| `--Re` | 6e6 | 雷诺数 |
| `--Mach` | 0.3 | 马赫数 |
| `--algo` | RL | 优化算法：`SA` 模拟退火 / `GA` 遗传算法 / `PSO` 粒子群 / `RL` 强化学习 |
| `--model` | kriging | 代理模型：`kriging` 高斯过程 / `nn` 神经网络 / `svr` 支持向量回归 |
| `--stop_after` | full | 停止阶段：`cst` / `sampling` / `xfoil` / `surrogate` / `full` |
| `-o, --output` | 同 dat_dir | 输出根目录 |
| `-q, --quiet` | False | 安静模式 |

### batch_run/config.py（底层全局配置）

修改此文件可调整更底层的参数：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `N1, N2` | 0.5, 1.0 | CST 类函数参数（圆头 N1=0.5，尖尾 N2=1.0） |
| `N_POINTS` | 201 | 翼型坐标点数 |
| `RANDOM_SEED` | 42 | 随机种子 |
| `FIX_DZ` | True | 是否固定尾缘 dz 项 |
| `T_MIN_THRESHOLD` | 1e-5 | 几何有效性：最小厚度阈值 |
| `X_MARGIN_CHECK` | 0.02 | 几何检查：前后缘忽略区域 |
| `ENFORCE_TMAX_NOT_SMALLER` | False | 是否强制采样厚度不低于基线 |

### youhua/main_optimization.py（优化算法参数）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `DELTA_AU/AL` | 0.065 | 优化搜索范围（应与采样范围一致） |
| `SHRINK_RATIO` | 0.05 | 搜索范围相对采样范围的收缩比例（防外推） |
| `PENALTY_CL` | 1e7 | Cl 约束违反罚函数系数 |
| `PENALTY_TMAX` | 1e7 | t_max 约束违反罚函数系数 |

#### GA 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `pop_size` | 150 | 种群大小 |
| `generations` | 500 | 迭代代数 |
| `crossover_rate` | 0.9 | 交叉概率 |
| `mutation_rate` | 0.15 | 变异概率 |
| `elite_size` | 4 | 精英保留数 |

#### SA 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_iter` | 6000 | 最大迭代次数 |
| `init_temp` | 1.0 | 初始温度 |
| `cooling_rate` | 0.997 | 降温速率（越接近 1 越慢） |
| `step_ratio` | 0.04 | 邻域扰动步长比例 |

#### PSO 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `swarm_size` | 120 | 粒子群大小 |
| `max_iter` | 500 | 最大迭代次数 |
| `w` | 0.72 | 惯性权重 |
| `c1, c2` | 1.7, 1.7 | 个体/社会学习因子 |

## 输出结构

```
multi_seed_output/seeds/
├── seed_cluster1_<翼型名>.dat
├── seed_cluster1_<翼型名>_output/
│   ├── seed_info.json            # 12 维 CST 参数 + 基线 Cl/Cd/t_max
│   ├── samples.csv               # LHS 采样结果
│   ├── surrogate_dataset.csv     # XFOIL 气动数据
│   ├── surrogate_saved_models/   # 代理模型 .pkl 文件
│   ├── optimization_results/
│   │   ├── data/
│   │   │   ├── *_best_result.json   # 最优结果
│   │   │   ├── *_summary.csv        # 汇总
│   │   │   └── xfoil_verify.json    # XFOIL 验证结果
│   │   ├── airfoils/                # 优化翼型 .dat
│   │   └── plots/                   # 收敛曲线、对比图
│   └── ...
├── seed_cluster2_<翼型名>.dat
├── seed_cluster2_<翼型名>_output/
└── ...
```

## 优化目标

最小化阻力系数 **Cd**，满足约束：

- Cl ≥ 基线升力系数
- t_max ≥ 基线最大厚度

约束通过罚函数处理（系数 1e7）。

## 注意事项

- 所有脚本的 `main()` 均支持参数化调用，也支持独立运行
- XFOIL 求解阶段耗时最长（每个种子 ~1500 次 XFOIL 调用）
- 未安装 PyTorch 时使用 `--algo GA`、`--algo SA` 或 `--algo PSO`
- `--n_samples` 建议 ≥ 200 以保证代理模型精度，1500 为推荐值
- 种子数 `-k` 建议 3~6，太多会显著增加总计算时间

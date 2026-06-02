# -*- coding: utf-8 -*-
"""
sample_size_study.py
--------------------
比较不同训练样本数对代理模型精度的影响。

用法：
1. 先确保 batch_run/surrogate_dataset.csv 已经由 XFOIL 批量计算生成。
2. 运行本脚本：
   python surrogate/sample_size_study.py

输出：
- surrogate/sample_size_study/sample_size_metrics.csv
- surrogate/sample_size_study/sample_size_metrics.xlsx
- surrogate/sample_size_study/plots/*.png

说明：
- 本脚本不会覆盖 surrogate_saved_models 里的正式模型。
- 测试集固定为全部有效数据的 15%，不同样本数只改变训练/验证池大小。
- 这样不同样本数之间的指标更可比。
"""

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from kriging_model import train_kriging, predict_kriging
from svr_model import train_svr, predict_svr
from nn_model import train_nn_with_curves, predict_nn


# =========================================================
# 一、实验配置
# =========================================================

ROOT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ROOT_DIR.parent
DATA_FILE = PROJECT_DIR / "batch_run" / "surrogate_dataset.csv"

SAVE_DIR = ROOT_DIR / "sample_size_study"
PLOT_DIR = SAVE_DIR / "plots"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "Au_0", "Au_1", "Au_2", "Au_3", "Au_4", "Au_5",
    "Al_0", "Al_1", "Al_2", "Al_3", "Al_4", "Al_5"
]

TARGETS = ["cl", "cd", "t_max"]
MODELS = ["kriging", "svr", "nn"]

# 从非测试数据池中抽取这些数量的样本用于 train + val。
# "all" 表示使用全部非测试数据。
TRAIN_POOL_SIZES = [100, 200, 400, 600, 800, "all"]

TEST_SIZE = 0.15
VAL_RATIO_IN_POOL = 0.15
RANDOM_STATE = 42

# 如果想降低单次实验偶然性，可以加更多种子，例如 [42, 7, 2026]。
# 注意：种子越多，训练次数越多。
REPEAT_SEEDS = [42]

# =========================================================
# 并行设置
# =========================================================
# scikit-learn 的 Kriging / SVR / MLPRegressor 主要是 CPU 训练；
# 这里用多进程并行跑不同实验组合，不会调用 GPU。
PARALLEL = True

# 建议先用 3 或 4。Kriging 比较吃 CPU/内存，开太大可能反而变慢。
MAX_WORKERS = 5


# =========================================================
# 二、模型超参数
# =========================================================

# 样本数实验会重复训练很多次，Kriging 优化重启次数设小一点以控制耗时。
# 最终正式训练仍可在 main_surrogate.py 中使用更大的 n_restarts。
KRIGING_KERNEL = "rbf"
KRIGING_NOISE_LEVEL = 1e-6
KRIGING_MATERN_NU = 1.5
KRIGING_N_RESTARTS = 5

SVR_KERNEL = "rbf"
SVR_C = 30.0
SVR_EPSILON = 0.001
SVR_GAMMA = "scale"
SVR_DEGREE = 3

NN_HIDDEN_LAYER_SIZES = (128, 64)
NN_ACTIVATION = "relu"
NN_ALPHA = 1e-4
NN_LR = 1e-3
NN_MAX_EPOCHS = 700
NN_PATIENCE = 50

NN_CONFIG_BY_TARGET = {
    "cl": {
        "hidden_layer_sizes": (128, 64),
        "activation": "relu",
        "alpha": 1e-4,
        "learning_rate_init": 1e-3,
        "max_epochs": 600,
        "patience": 60,
    },
    "cd": {
        "hidden_layer_sizes": (256, 128),
        "activation": "relu",
        "alpha": 3e-2,
        "learning_rate_init": 5e-4,
        "max_epochs": 800,
        "patience": 80,
    },
    "t_max": {
        "hidden_layer_sizes": (256, 128),
        "activation": "relu",
        "alpha": 1e-4,
        "learning_rate_init": 1e-3,
        "max_epochs": 700,
        "patience": 70,
    },
}

TARGET_TRANSFORM_BY_MODEL_TARGET = {
    ("nn", "cd"): "log",
}


# =========================================================
# 三、工具函数
# =========================================================

def calc_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-12))) * 100.0
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": rmse,
        "mae": mae,
        "mape_percent": mape,
    }


def get_nn_config(target):
    config = {
        "hidden_layer_sizes": NN_HIDDEN_LAYER_SIZES,
        "activation": NN_ACTIVATION,
        "alpha": NN_ALPHA,
        "learning_rate_init": NN_LR,
        "max_epochs": NN_MAX_EPOCHS,
        "patience": NN_PATIENCE,
    }
    config.update(NN_CONFIG_BY_TARGET.get(target, {}))
    return config


def get_target_transform(model_name, target):
    return TARGET_TRANSFORM_BY_MODEL_TARGET.get(
        (model_name.lower(), target),
        "identity",
    )


def apply_target_transform(y, transform):
    y = np.asarray(y, dtype=float)
    if transform == "identity":
        return y
    if transform == "log":
        if np.any(y <= 0):
            raise ValueError("log target transform requires all y > 0")
        return np.log(y)
    raise ValueError(f"Unsupported target transform: {transform}")


def inverse_target_transform(y, transform):
    y = np.asarray(y, dtype=float)
    if transform == "identity":
        return y
    if transform == "log":
        return np.exp(y)
    raise ValueError(f"Unsupported target transform: {transform}")


def split_scaled_data(df, target, train_pool_size, subset_seed, model_name):
    needed_cols = FEATURE_COLS + [target]
    missing_cols = [c for c in needed_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"数据文件缺少列: {missing_cols}")

    X = df[FEATURE_COLS].values
    y_raw = df[target].values
    target_transform = get_target_transform(model_name, target)
    y_model = apply_target_transform(y_raw, target_transform)

    X_pool, X_test, y_pool, y_test, y_raw_pool, y_raw_test = train_test_split(
        X,
        y_model,
        y_raw,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    if train_pool_size == "all":
        X_use = X_pool
        y_use = y_pool
        y_raw_use = y_raw_pool
        n_pool = len(X_pool)
    else:
        n_pool = min(int(train_pool_size), len(X_pool))
        rng = np.random.default_rng(subset_seed)
        idx = rng.choice(len(X_pool), size=n_pool, replace=False)
        X_use = X_pool[idx]
        y_use = y_pool[idx]
        y_raw_use = y_raw_pool[idx]

    X_train, X_val, y_train, y_val, y_raw_train, y_raw_val = train_test_split(
        X_use,
        y_use,
        y_raw_use,
        test_size=VAL_RATIO_IN_POOL,
        random_state=subset_seed,
    )

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_scaled = x_scaler.fit_transform(X_train)
    X_val_scaled = x_scaler.transform(X_val)
    X_test_scaled = x_scaler.transform(X_test)

    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).ravel()

    return {
        "n_pool": n_pool,
        "X_train_scaled": X_train_scaled,
        "X_val_scaled": X_val_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train_scaled": y_train_scaled,
        "y_val_scaled": y_val_scaled,
        "y_test": y_raw_test,
        "y_scaler": y_scaler,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "target": target,
        "target_transform": target_transform,
    }


def train_and_predict(model_name, data):
    if model_name == "kriging":
        model = train_kriging(
            data["X_train_scaled"],
            data["y_train_scaled"],
            kernel_name=KRIGING_KERNEL,
            noise_level=KRIGING_NOISE_LEVEL,
            matern_nu=KRIGING_MATERN_NU,
            n_restarts_optimizer=KRIGING_N_RESTARTS,
            random_state=RANDOM_STATE,
        )
        return predict_kriging(model, data["X_test_scaled"])

    if model_name == "svr":
        model = train_svr(
            data["X_train_scaled"],
            data["y_train_scaled"],
            kernel=SVR_KERNEL,
            C=SVR_C,
            epsilon=SVR_EPSILON,
            gamma=SVR_GAMMA,
            degree=SVR_DEGREE,
        )
        return predict_svr(model, data["X_test_scaled"])

    if model_name == "nn":
        nn_config = get_nn_config(data["target"])
        model, _ = train_nn_with_curves(
            data["X_train_scaled"],
            data["y_train_scaled"],
            data["X_val_scaled"],
            data["y_val_scaled"],
            hidden_layer_sizes=nn_config["hidden_layer_sizes"],
            activation=nn_config["activation"],
            alpha=nn_config["alpha"],
            learning_rate_init=nn_config["learning_rate_init"],
            max_epochs=nn_config["max_epochs"],
            patience=nn_config["patience"],
            random_state=RANDOM_STATE,
        )
        return predict_nn(model, data["X_test_scaled"])

    raise ValueError(f"未知模型: {model_name}")


def load_valid_dataset():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"找不到数据文件: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    df = df[df["status"] == "ok"].copy()

    if len(df) == 0:
        raise ValueError("surrogate_dataset.csv 中没有 status == 'ok' 的有效样本")

    return df


def evaluate_case(df, target, train_pool_size, repeat_seed, model_name):
    data = split_scaled_data(
        df=df,
        target=target,
        train_pool_size=train_pool_size,
        subset_seed=repeat_seed,
        model_name=model_name,
    )

    print(
        f"target={target}, model={model_name}, "
        f"n_pool={data['n_pool']}, seed={repeat_seed}"
    )

    try:
        y_pred_scaled = train_and_predict(model_name, data)
        y_pred_model_space = data["y_scaler"].inverse_transform(
            np.asarray(y_pred_scaled).reshape(-1, 1)
        ).ravel()
        y_pred = inverse_target_transform(
            y_pred_model_space,
            data["target_transform"],
        )
        metrics = calc_metrics(data["y_test"], y_pred)
        nn_config = get_nn_config(target) if model_name == "nn" else {}
        return {
            "model": model_name,
            "target": target,
            "target_transform": data["target_transform"],
            "requested_n_pool": train_pool_size,
            "n_pool": data["n_pool"],
            "n_train": data["n_train"],
            "n_val": data["n_val"],
            "n_test": data["n_test"],
            "seed": repeat_seed,
            **metrics,
            "nn_hidden_layer_sizes": str(nn_config.get("hidden_layer_sizes", "")),
            "nn_activation": nn_config.get("activation", ""),
            "nn_alpha": nn_config.get("alpha", ""),
            "nn_learning_rate_init": nn_config.get("learning_rate_init", ""),
            "nn_max_epochs": nn_config.get("max_epochs", ""),
            "nn_patience": nn_config.get("patience", ""),
            "status": "ok",
            "error": "",
        }
    except Exception as e:
        print(f"运行失败: {model_name}, {target}, n={data['n_pool']}: {e}")
        return {
            "model": model_name,
            "target": target,
            "target_transform": data["target_transform"],
            "requested_n_pool": train_pool_size,
            "n_pool": data["n_pool"],
            "n_train": data["n_train"],
            "n_val": data["n_val"],
            "n_test": data["n_test"],
            "seed": repeat_seed,
            "r2": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "mape_percent": np.nan,
            "nn_hidden_layer_sizes": "",
            "nn_activation": "",
            "nn_alpha": "",
            "nn_learning_rate_init": "",
            "nn_max_epochs": "",
            "nn_patience": "",
            "status": "failed",
            "error": str(e),
        }


def evaluate_case_worker(task):
    target, train_pool_size, repeat_seed, model_name = task
    df = load_valid_dataset()
    return evaluate_case(df, target, train_pool_size, repeat_seed, model_name)


def build_tasks():
    tasks = []
    for target in TARGETS:
        for train_pool_size in TRAIN_POOL_SIZES:
            for repeat_seed in REPEAT_SEEDS:
                for model_name in MODELS:
                    tasks.append((target, train_pool_size, repeat_seed, model_name))

    # 调试用：PowerShell 中设置 $env:SAMPLE_STUDY_MAX_TASKS='6'
    max_tasks = os.environ.get("SAMPLE_STUDY_MAX_TASKS")
    if max_tasks:
        tasks = tasks[:int(max_tasks)]
        print(f"调试模式：仅运行前 {len(tasks)} 个实验任务")

    return tasks


def run_serial(tasks):
    df = load_valid_dataset()
    return [
        evaluate_case(df, target, train_pool_size, repeat_seed, model_name)
        for target, train_pool_size, repeat_seed, model_name in tasks
    ]


def run_parallel(tasks):
    max_workers = min(MAX_WORKERS, len(tasks))
    print(f"并行模式：MAX_WORKERS = {max_workers}, tasks = {len(tasks)}")

    rows = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(evaluate_case_worker, task): task
            for task in tasks
        }

        for i, future in enumerate(as_completed(future_map), start=1):
            task = future_map[future]
            try:
                row = future.result()
            except Exception as e:
                target, train_pool_size, repeat_seed, model_name = task
                row = {
                    "model": model_name,
                    "target": target,
                    "target_transform": get_target_transform(model_name, target),
                    "requested_n_pool": train_pool_size,
                    "n_pool": np.nan,
                    "n_train": np.nan,
                    "n_val": np.nan,
                    "n_test": np.nan,
                    "seed": repeat_seed,
                    "r2": np.nan,
                    "rmse": np.nan,
                    "mae": np.nan,
                    "mape_percent": np.nan,
                    "nn_hidden_layer_sizes": "",
                    "nn_activation": "",
                    "nn_alpha": "",
                    "nn_learning_rate_init": "",
                    "nn_max_epochs": "",
                    "nn_patience": "",
                    "status": "failed",
                    "error": str(e),
                }

            rows.append(row)
            print(
                f"[{i}/{len(tasks)}] {row['model']} {row['target']} "
                f"n={row['requested_n_pool']} status={row['status']}"
            )

    return rows


def plot_metric(summary_df, target, metric):
    plt.figure(figsize=(8, 5))
    plotted = 0

    for model in MODELS:
        part = summary_df[
            (summary_df["target"] == target)
            & (summary_df["model"] == model)
        ].sort_values("n_pool")
        if part.empty:
            continue

        plt.plot(
            part["n_pool"],
            part[metric],
            marker="o",
            linewidth=2,
            label=model,
        )
        plotted += 1

    plt.xlabel("Training + validation sample count")
    plt.ylabel(metric)
    plt.title(f"Sample size effect - {target} - {metric}")
    plt.grid(True)
    if plotted > 0:
        plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{target}_{metric}.png", dpi=300)
    plt.close()


def save_outputs(rows):
    result_df = pd.DataFrame(rows)
    csv_path = SAVE_DIR / "sample_size_metrics.csv"
    xlsx_path = SAVE_DIR / "sample_size_metrics.xlsx"

    result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    group_cols = ["model", "target", "n_pool"]
    summary_df = (
        result_df
        .groupby(group_cols, as_index=False)
        .agg(
            target_transform=("target_transform", "first"),
            n_train=("n_train", "mean"),
            n_val=("n_val", "mean"),
            n_test=("n_test", "mean"),
            r2=("r2", "mean"),
            rmse=("rmse", "mean"),
            mae=("mae", "mean"),
            mape_percent=("mape_percent", "mean"),
            r2_std=("r2", "std"),
            rmse_std=("rmse", "std"),
        )
    )

    config_rows = [
        ["DATA_FILE", str(DATA_FILE)],
        ["TRAIN_POOL_SIZES", str(TRAIN_POOL_SIZES)],
        ["TEST_SIZE", TEST_SIZE],
        ["VAL_RATIO_IN_POOL", VAL_RATIO_IN_POOL],
        ["RANDOM_STATE", RANDOM_STATE],
        ["REPEAT_SEEDS", str(REPEAT_SEEDS)],
        ["KRIGING_KERNEL", KRIGING_KERNEL],
        ["KRIGING_NOISE_LEVEL", KRIGING_NOISE_LEVEL],
        ["KRIGING_MATERN_NU", KRIGING_MATERN_NU],
        ["KRIGING_N_RESTARTS", KRIGING_N_RESTARTS],
        ["SVR_KERNEL", SVR_KERNEL],
        ["SVR_C", SVR_C],
        ["SVR_EPSILON", SVR_EPSILON],
        ["SVR_GAMMA", SVR_GAMMA],
        ["SVR_DEGREE", SVR_DEGREE],
        ["NN_HIDDEN_LAYER_SIZES", str(NN_HIDDEN_LAYER_SIZES)],
        ["NN_ACTIVATION", NN_ACTIVATION],
        ["NN_ALPHA", NN_ALPHA],
        ["NN_LR", NN_LR],
        ["NN_MAX_EPOCHS", NN_MAX_EPOCHS],
        ["NN_PATIENCE", NN_PATIENCE],
        ["NN_CONFIG_BY_TARGET", str(NN_CONFIG_BY_TARGET)],
        ["TARGET_TRANSFORM_BY_MODEL_TARGET", str(TARGET_TRANSFORM_BY_MODEL_TARGET)],
        ["PARALLEL", PARALLEL],
        ["MAX_WORKERS", MAX_WORKERS],
    ]
    config_df = pd.DataFrame(config_rows, columns=["parameter", "value"])

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        result_df.to_excel(writer, sheet_name="all_runs", index=False)
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        config_df.to_excel(writer, sheet_name="config", index=False)

    for target in TARGETS:
        for metric in ["rmse", "r2", "mape_percent"]:
            plot_metric(summary_df, target, metric)

    return csv_path, xlsx_path


# =========================================================
# 四、主程序
# =========================================================

def main():
    global SAVE_DIR, PLOT_DIR

    max_tasks = os.environ.get("SAMPLE_STUDY_MAX_TASKS")
    if max_tasks:
        SAVE_DIR = ROOT_DIR / "sample_size_study" / f"debug_{max_tasks}"
        PLOT_DIR = SAVE_DIR / "plots"
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        PLOT_DIR.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks()
    if PARALLEL:
        rows = run_parallel(tasks)
    else:
        rows = run_serial(tasks)

    csv_path, xlsx_path = save_outputs(rows)

    print("=" * 60)
    print("样本数影响实验完成")
    print(f"结果 CSV : {csv_path}")
    print(f"结果 Excel: {xlsx_path}")
    print(f"趋势图目录: {PLOT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

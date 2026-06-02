import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from kriging_model import train_kriging, predict_kriging
from svr_model import train_svr, predict_svr
from nn_model import train_nn_with_curves, predict_nn


# =========================================================
# 路径设置：以当前 .py 文件所在文件夹为根目录
# 假设当前代码在 surrogate 文件夹中
ROOT_DIR = Path(__file__).resolve().parent

# 总目录
PROJECT_DIR = ROOT_DIR.parent

# 1. 数据文件路径
DATA_FILE = PROJECT_DIR / "batch_run" / "surrogate_dataset.csv"

# 2. 图片保存路径
FIG_SAVE_DIR = ROOT_DIR / "surrogate_plots"
FIG_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 3. 模型保存路径
MODEL_SAVE_DIR = ROOT_DIR / "surrogate_saved_models"
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 4. 汇总 Excel 路径
SUMMARY_EXCEL_PATH = ROOT_DIR / "surrogate_summary.xlsx"

# 5. 是否保存模型
SAVE_MODEL = True

# 6. 运行模式：
# "single" = 单独跑一个模型 + 一个目标变量
# "all"    = 一次性跑完全部模型 + 全部目标变量
RUN_MODE = "all"

# 7. 单独模式下使用
MODEL_NAME = "nn"      # "kriging" / "svr" / "nn"
TARGET_COL = "cd"      # "cl" / "cd" / "t_max"

# 8. 全跑模式下使用
ALL_MODELS = ["kriging", "svr", "nn"]
ALL_TARGETS = ["cl", "cd", "t_max"]

# 9. 输入特征
FEATURE_COLS = [
    "Au_0", "Au_1", "Au_2", "Au_3", "Au_4", "Au_5",
    "Al_0", "Al_1", "Al_2", "Al_3", "Al_4", "Al_5"
]

# 10. 数据划分
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42

# 11. 是否自动打开图片
# 并行训练时不建议自动打开图片，否则会同时弹出很多窗口。
AUTO_OPEN_PLOTS = False

# 12. Kriging 是否限制训练样本数（可选）
LIMIT_KRIGING_SAMPLES = False
MAX_KRIGING_SAMPLES = 300

# 13. CPU 并行设置
# scikit-learn 这些模型主要使用 CPU。这里并行的是不同的“模型-目标”实验。
PARALLEL = True
MAX_WORKERS = 4


# =========================================================
# 二、Kriging 超参数
# =========================================================
KRIGING_KERNEL = "rbf"      # "rbf" / "matern" / "rq"
KRIGING_NOISE_LEVEL = 1e-6
KRIGING_MATERN_NU = 1.5
KRIGING_N_RESTARTS = 5


# =========================================================
# 三、SVR 超参数
# =========================================================
SVR_KERNEL = "rbf"          # "rbf" / "linear" / "poly" / "sigmoid"
SVR_C = 30.0
SVR_EPSILON = 0.001
SVR_GAMMA = "scale"
SVR_DEGREE = 3


# =========================================================
# 四、神经网络超参数
# =========================================================
# 默认 NN 参数；如果目标变量在 NN_CONFIG_BY_TARGET 里，会优先使用目标专属配置。
NN_HIDDEN_LAYER_SIZES = (128, 64)
NN_ACTIVATION = "relu"      # "relu" / "tanh" / "logistic" / "identity"
NN_ALPHA = 1e-4
NN_LR = 1e-3
NN_MAX_EPOCHS = 700
NN_PATIENCE = 50

# 不同输出量的数据规律不一样，NN 不强制共用同一套网络。
# 当前经验：
# - cl 用较小网络更稳，避免过拟合；
# - cd 使用 log(cd) 训练后，较强正则的两层网络更稳；
# - t_max 几何量相对平滑，两层增强网络即可。
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
        "learning_rate_init":5e-4,
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

# 目标变量变换。cd 始终为正，使用 log(cd) 训练通常更适合控制相对误差。
# 保存模型时会同时保存 meta.json，优化程序会根据 meta 自动 exp 还原。
TARGET_TRANSFORM_BY_MODEL_TARGET = {
    ("nn", "cd"): "log",
}

# =========================================================
# 工具函数
# =========================================================
def open_image_if_needed(image_path):
    if AUTO_OPEN_PLOTS:
        try:
            os.startfile(str(image_path))
        except Exception as e:
            print(f"自动打开图片失败: {image_path}")
            print(e)


def calc_metrics(y_true, y_pred):
    """
    在原始物理尺度下计算指标
    """
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    eps = 1e-12
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0

    return {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "mape": mape
    }


def get_nn_config(target_col):
    """返回指定目标变量的 NN 配置。"""
    default_config = {
        "hidden_layer_sizes": NN_HIDDEN_LAYER_SIZES,
        "activation": NN_ACTIVATION,
        "alpha": NN_ALPHA,
        "learning_rate_init": NN_LR,
        "max_epochs": NN_MAX_EPOCHS,
        "patience": NN_PATIENCE,
    }
    config = default_config.copy()
    config.update(NN_CONFIG_BY_TARGET.get(target_col, {}))
    return config


def get_target_transform(model_name, target_col):
    return TARGET_TRANSFORM_BY_MODEL_TARGET.get(
        (model_name.lower(), target_col),
        "identity"
    )


def apply_target_transform(y, transform):
    y = np.asarray(y, dtype=float)
    if transform == "identity":
        return y
    if transform == "log":
        if np.any(y <= 0):
            raise ValueError("log 目标变换要求 y 全部大于 0")
        return np.log(y)
    raise ValueError(f"不支持的目标变换: {transform}")


def inverse_target_transform(y, transform):
    y = np.asarray(y, dtype=float)
    if transform == "identity":
        return y
    if transform == "log":
        return np.exp(y)
    raise ValueError(f"不支持的目标变换: {transform}")


def plot_true_vs_pred(y_true, y_pred, model_name, target_name, metrics, save_path):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.75)

    y_min = min(np.min(y_true), np.min(y_pred))
    y_max = max(np.max(y_true), np.max(y_pred))
    plt.plot([y_min, y_max], [y_min, y_max], "r--", lw=2)

    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"{model_name.upper()} - {target_name}")
    plt.grid(True)

    textstr = (
        f"R²   = {metrics['r2']:.4f}\n"
        f"RMSE = {metrics['rmse']:.6f}\n"
        f"MAE  = {metrics['mae']:.6f}\n"
        f"MAPE = {metrics['mape']:.2f}%"
    )

    plt.text(
        0.05, 0.95, textstr,
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    open_image_if_needed(save_path)


def plot_nn_loss_curve(history, target_name, save_path):
    plt.figure(figsize=(7, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (scaled space)")
    plt.title(f"NN Loss Curve - {target_name}")
    plt.grid(True)
    plt.legend()

    textstr = (
        f"Epochs = {history['epochs']}\n"
        f"Best Val Loss = {history['best_val_loss']:.6f}"
    )

    plt.text(
        0.60, 0.95, textstr,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    open_image_if_needed(save_path)


def prepare_data(df, target_col, model_name):
    """
    提取数据并做 train / val / test 划分与归一化
    """
    needed_cols = FEATURE_COLS + [target_col]
    missing_cols = [c for c in needed_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"数据文件缺少列: {missing_cols}")

    df_use = df.copy()

    if model_name.lower() == "kriging" and LIMIT_KRIGING_SAMPLES:
        df_use = df_use.iloc[:MAX_KRIGING_SAMPLES].copy()

    X = df_use[FEATURE_COLS].values
    y_raw = df_use[target_col].values
    target_transform = get_target_transform(model_name, target_col)
    y_model = apply_target_transform(y_raw, target_transform)

    X_temp, X_test, y_temp, y_test, y_raw_temp, y_raw_test = train_test_split(
        X, y_model, y_raw,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    val_ratio_in_temp = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val, y_raw_train, y_raw_val = train_test_split(
        X_temp, y_temp, y_raw_temp,
        test_size=val_ratio_in_temp,
        random_state=RANDOM_STATE
    )

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_scaled = x_scaler.fit_transform(X_train)
    X_val_scaled = x_scaler.transform(X_val)
    X_test_scaled = x_scaler.transform(X_test)

    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).ravel()
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

    return {
        "df_use": df_use,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_raw_train,
        "y_val": y_raw_val,
        "y_test": y_raw_test,
        "X_train_scaled": X_train_scaled,
        "X_val_scaled": X_val_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train_scaled": y_train_scaled,
        "y_val_scaled": y_val_scaled,
        "y_test_scaled": y_test_scaled,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "target_transform": target_transform,
    }


def save_model_bundle(model, x_scaler, y_scaler, model_name, target_col, target_transform="identity"):
    if not SAVE_MODEL or model is None:
        return None

    model_file = MODEL_SAVE_DIR / f"{model_name}_{target_col}.pkl"
    x_scaler_file = MODEL_SAVE_DIR / f"{model_name}_{target_col}_x_scaler.pkl"
    y_scaler_file = MODEL_SAVE_DIR / f"{model_name}_{target_col}_y_scaler.pkl"
    meta_file = MODEL_SAVE_DIR / f"{model_name}_{target_col}_meta.json"

    joblib.dump(model, model_file)
    joblib.dump(x_scaler, x_scaler_file)
    joblib.dump(y_scaler, y_scaler_file)

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": model_name,
                "target_col": target_col,
                "target_transform": target_transform,
            },
            f,
            ensure_ascii=False,
            indent=4
        )

    return {
        "model_file": str(model_file),
        "x_scaler_file": str(x_scaler_file),
        "y_scaler_file": str(y_scaler_file),
        "meta_file": str(meta_file),
    }


def run_one_experiment(df, model_name, target_col):
    """
    运行一组实验：一个模型 + 一个目标变量
    返回结果字典
    """
    print("=" * 60)
    print(f"开始运行: model={model_name}, target={target_col}")
    print("=" * 60)

    data = prepare_data(df, target_col, model_name)

    model = None
    y_pred_scaled = None
    history = None

    # -----------------------------------------------------
    # 模型训练
    # -----------------------------------------------------
    if model_name.lower() == "kriging":
        model = train_kriging(
            data["X_train_scaled"], data["y_train_scaled"],
            kernel_name=KRIGING_KERNEL,
            noise_level=KRIGING_NOISE_LEVEL,
            matern_nu=KRIGING_MATERN_NU,
            n_restarts_optimizer=KRIGING_N_RESTARTS,
            random_state=RANDOM_STATE
        )
        y_pred_scaled = predict_kriging(model, data["X_test_scaled"])

    elif model_name.lower() == "svr":
        model = train_svr(
            data["X_train_scaled"], data["y_train_scaled"],
            kernel=SVR_KERNEL,
            C=SVR_C,
            epsilon=SVR_EPSILON,
            gamma=SVR_GAMMA,
            degree=SVR_DEGREE
        )
        y_pred_scaled = predict_svr(model, data["X_test_scaled"])

    elif model_name.lower() == "nn":
        nn_config = get_nn_config(target_col)
        model, history = train_nn_with_curves(
            data["X_train_scaled"], data["y_train_scaled"],
            data["X_val_scaled"], data["y_val_scaled"],
            hidden_layer_sizes=nn_config["hidden_layer_sizes"],
            activation=nn_config["activation"],
            alpha=nn_config["alpha"],
            learning_rate_init=nn_config["learning_rate_init"],
            max_epochs=nn_config["max_epochs"],
            patience=nn_config["patience"],
            random_state=RANDOM_STATE
        )
        y_pred_scaled = predict_nn(model, data["X_test_scaled"])

        loss_fig_path = FIG_SAVE_DIR / f"NN_loss_curve_{target_col}.png"
        plot_nn_loss_curve(history, target_col, loss_fig_path)

    else:
        raise ValueError("model_name 只能是 'kriging'、'svr' 或 'nn'。")

    # -----------------------------------------------------
    # 模型保存
    # -----------------------------------------------------
    save_info = save_model_bundle(
        model,
        data["x_scaler"],
        data["y_scaler"],
        model_name,
        target_col,
        target_transform=data["target_transform"]
    )

    # -----------------------------------------------------
    # 反归一化与测试集评估
    # -----------------------------------------------------
    y_pred_model_space = data["y_scaler"].inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_pred = inverse_target_transform(y_pred_model_space, data["target_transform"])
    metrics = calc_metrics(data["y_test"], y_pred)

    print(f"测试集结果 - {model_name} - {target_col}")
    print(f"R²   = {metrics['r2']:.6f}")
    print(f"RMSE = {metrics['rmse']:.6f}")
    print(f"MAE  = {metrics['mae']:.6f}")
    print(f"MAPE = {metrics['mape']:.2f}%")

    # -----------------------------------------------------
    # 结果图
    # -----------------------------------------------------
    result_fig_path = FIG_SAVE_DIR / f"{model_name}_{target_col}_true_vs_pred.png"
    plot_true_vs_pred(
        data["y_test"], y_pred,
        model_name, target_col,
        metrics,
        result_fig_path
    )

    result = {
        "model": model_name,
        "target": target_col,
        "n_samples": len(data["df_use"]),
        "n_train": len(data["X_train"]),
        "n_val": len(data["X_val"]),
        "n_test": len(data["X_test"]),
        "r2": metrics["r2"],
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "mape_percent": metrics["mape"],
        "result_fig": str(result_fig_path),
        "loss_fig": str(FIG_SAVE_DIR / f"NN_loss_curve_{target_col}.png") if model_name.lower() == "nn" else "",
        "model_file": save_info["model_file"] if save_info else "",
        "x_scaler_file": save_info["x_scaler_file"] if save_info else "",
        "y_scaler_file": save_info["y_scaler_file"] if save_info else "",
        "meta_file": save_info["meta_file"] if save_info else "",
        "target_transform": data["target_transform"],
    }

    if model_name.lower() == "nn":
        result.update({
            "nn_hidden_layer_sizes": str(nn_config["hidden_layer_sizes"]),
            "nn_activation": nn_config["activation"],
            "nn_alpha": nn_config["alpha"],
            "nn_learning_rate_init": nn_config["learning_rate_init"],
            "nn_max_epochs": nn_config["max_epochs"],
            "nn_patience": nn_config["patience"],
            "nn_epochs_used": history["epochs"] if history else "",
            "nn_best_val_loss": history["best_val_loss"] if history else "",
        })

    return result


def save_summary_excel(results):
    """
    全跑模式下，把结果保存到一个 Excel
    """
    summary_df = pd.DataFrame(results)

    config_rows = [
        ["DATA_FILE", str(DATA_FILE)],
        ["FIG_SAVE_DIR", str(FIG_SAVE_DIR)],
        ["MODEL_SAVE_DIR", str(MODEL_SAVE_DIR)],
        ["SAVE_MODEL", SAVE_MODEL],
        ["RUN_MODE", RUN_MODE],
        ["TEST_SIZE", TEST_SIZE],
        ["VAL_SIZE", VAL_SIZE],
        ["RANDOM_STATE", RANDOM_STATE],
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
        ["LIMIT_KRIGING_SAMPLES", LIMIT_KRIGING_SAMPLES],
        ["MAX_KRIGING_SAMPLES", MAX_KRIGING_SAMPLES],
    ]
    config_df = pd.DataFrame(config_rows, columns=["parameter", "value"])

    with pd.ExcelWriter(SUMMARY_EXCEL_PATH, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        config_df.to_excel(writer, sheet_name="config", index=False)

    print(f"汇总 Excel 已保存: {SUMMARY_EXCEL_PATH}")


def load_training_dataframe():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"找不到数据文件: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    df = df[df["status"] == "ok"].copy()

    if len(df) == 0:
        raise ValueError("数据文件中没有 status == 'ok' 的有效样本")

    return df


def run_one_experiment_worker(task):
    model_name, target_col = task
    df = load_training_dataframe()
    return run_one_experiment(df, model_name, target_col)


def build_all_tasks():
    tasks = []
    for model_name in ALL_MODELS:
        for target_col in ALL_TARGETS:
            tasks.append((model_name, target_col))

    # 调试用：PowerShell 中设置 $env:SURROGATE_MAX_TASKS='2'
    max_tasks = os.environ.get("SURROGATE_MAX_TASKS")
    if max_tasks:
        tasks = tasks[:int(max_tasks)]
        print(f"调试模式：仅运行前 {len(tasks)} 个训练任务")

    return tasks


def make_failed_result(model_name, target_col, error):
    return {
        "model": model_name,
        "target": target_col,
        "n_samples": np.nan,
        "n_train": np.nan,
        "n_val": np.nan,
        "n_test": np.nan,
        "r2": np.nan,
        "rmse": np.nan,
        "mae": np.nan,
        "mape_percent": np.nan,
        "result_fig": "",
        "loss_fig": "",
        "model_file": "",
        "x_scaler_file": "",
        "y_scaler_file": "",
        "error": str(error),
    }


def run_all_experiments_serial(tasks):
    df = load_training_dataframe()
    results = []

    for model_name, target_col in tasks:
        try:
            results.append(run_one_experiment(df, model_name, target_col))
        except Exception as e:
            print(f"运行失败: model={model_name}, target={target_col}")
            print(e)
            results.append(make_failed_result(model_name, target_col, e))

    return results


def run_all_experiments_parallel(tasks):
    max_workers = min(MAX_WORKERS, len(tasks))
    print(f"并行模式：MAX_WORKERS = {max_workers}, tasks = {len(tasks)}")

    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(run_one_experiment_worker, task): task
            for task in tasks
        }

        for i, future in enumerate(as_completed(future_map), start=1):
            model_name, target_col = future_map[future]
            try:
                result = future.result()
            except Exception as e:
                print(f"运行失败: model={model_name}, target={target_col}")
                print(e)
                result = make_failed_result(model_name, target_col, e)

            results.append(result)
            print(
                f"[{i}/{len(tasks)}] model={model_name}, "
                f"target={target_col}, r2={result.get('r2', np.nan)}"
            )

    order = {
        (model_name, target_col): i
        for i, (model_name, target_col) in enumerate(tasks)
    }
    results.sort(key=lambda item: order.get((item["model"], item["target"]), 10**12))
    return results


# =========================================================
# 主程序入口
# =========================================================
def main(data_file=None, model_save_dir=None, fig_save_dir=None,
         summary_path=None):
    """
    代理模型训练主函数。

    参数（全部可选，默认使用模块级硬编码值）:
        data_file:       surrogate_dataset.csv 路径
        model_save_dir:  模型 .pkl 保存目录
        fig_save_dir:    评估图表保存目录
        summary_path:    汇总 Excel 路径（仅 all 模式）
    """
    global DATA_FILE, MODEL_SAVE_DIR, FIG_SAVE_DIR, SUMMARY_EXCEL_PATH
    _orig_data = DATA_FILE
    _orig_model = MODEL_SAVE_DIR
    _orig_fig = FIG_SAVE_DIR
    _orig_summary = SUMMARY_EXCEL_PATH

    if data_file is not None:
        DATA_FILE = Path(data_file)
    if model_save_dir is not None:
        MODEL_SAVE_DIR = Path(model_save_dir)
    if fig_save_dir is not None:
        FIG_SAVE_DIR = Path(fig_save_dir)
    if summary_path is not None:
        SUMMARY_EXCEL_PATH = Path(summary_path)

    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        if RUN_MODE.lower() == "single":
            df = load_training_dataframe()
            result = run_one_experiment(df, MODEL_NAME, TARGET_COL)

            print("=" * 60)
            print("单次运行完成")
            print(f"模型: {result['model']}")
            print(f"目标: {result['target']}")
            print(f"R²: {result['r2']:.6f}")
            print(f"RMSE: {result['rmse']:.6f}")
            print(f"MAE: {result['mae']:.6f}")
            print(f"MAPE: {result['mape_percent']:.2f}%")
            print(f"图片保存路径: {FIG_SAVE_DIR}")
            print(f"模型保存路径: {MODEL_SAVE_DIR}")
            print("=" * 60)

        elif RUN_MODE.lower() == "all":
            tasks = build_all_tasks()
            if PARALLEL:
                all_results = run_all_experiments_parallel(tasks)
            else:
                all_results = run_all_experiments_serial(tasks)

            save_summary_excel(all_results)

            print("=" * 60)
            print("全量运行完成")
            print(f"图片保存路径: {FIG_SAVE_DIR}")
            print(f"模型保存路径: {MODEL_SAVE_DIR}")
            print(f"汇总Excel路径: {SUMMARY_EXCEL_PATH}")
            print("=" * 60)

            return all_results

        else:
            raise ValueError("RUN_MODE 只能是 'single' 或 'all'。")
    finally:
        DATA_FILE = _orig_data
        MODEL_SAVE_DIR = _orig_model
        FIG_SAVE_DIR = _orig_fig
        SUMMARY_EXCEL_PATH = _orig_summary


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

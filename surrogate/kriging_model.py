import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel, RBF, Matern, RationalQuadratic, WhiteKernel
)


def build_kernel(kernel_name: str, input_dim: int,
                 noise_level: float = 1e-6,
                 matern_nu: float = 1.5):
    kernel_name = kernel_name.lower()

    if kernel_name == "rbf":
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * RBF(length_scale=np.ones(input_dim), length_scale_bounds=(1e-3, 1e3))
            + WhiteKernel(noise_level=noise_level, noise_level_bounds=(1e-10, 1e-2))
        )
    elif kernel_name == "matern":
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(length_scale=np.ones(input_dim), length_scale_bounds=(1e-3, 1e3), nu=matern_nu)
            + WhiteKernel(noise_level=noise_level, noise_level_bounds=(1e-10, 1e-2))
        )
    elif kernel_name == "rq":
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * RationalQuadratic(length_scale=1.0, alpha=1.0)
            + WhiteKernel(noise_level=noise_level, noise_level_bounds=(1e-10, 1e-2))
        )
    else:
        raise ValueError(f"不支持的 Kriging kernel_name: {kernel_name}")

    return kernel


def train_kriging(X_train, y_train,
                  kernel_name="rbf",
                  noise_level=1e-6,
                  matern_nu=1.5,
                  n_restarts_optimizer=5,
                  random_state=42):
    kernel = build_kernel(
        kernel_name=kernel_name,
        input_dim=X_train.shape[1],
        noise_level=noise_level,
        matern_nu=matern_nu
    )

    model = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=n_restarts_optimizer,
        normalize_y=False,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def predict_kriging(model, X):
    return model.predict(X)
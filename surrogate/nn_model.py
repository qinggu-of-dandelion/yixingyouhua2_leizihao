import copy
import warnings
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def train_nn_with_curves(
    X_train, y_train,
    X_val, y_val,
    hidden_layer_sizes=(64, 64),
    activation="relu",
    alpha=1e-4,
    learning_rate_init=1e-3,
    max_epochs=500,
    patience=50,
    random_state=42
):
    #上面是默认值，要改的话直接在主函数里面改就好
    """
    用 sklearn 的 MLPRegressor + warm_start 模拟逐 epoch 训练，
    输出训练损失和验证损失曲线。
    """
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver="adam",
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=1,      # 每次只训练1个epoch
        warm_start=True,
        shuffle=True,
        random_state=random_state
    )

    train_losses = []
    val_losses = []

    best_val_loss = np.inf
    best_model = None
    wait = 0

    for epoch in range(max_epochs):
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        train_loss = mean_squared_error(y_train, y_train_pred)
        val_loss = mean_squared_error(y_val, y_val_pred)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            break

    history = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "best_val_loss": best_val_loss,
        "epochs": len(train_losses)
    }

    return best_model, history


def predict_nn(model, X):
    return model.predict(X)
from sklearn.svm import SVR


def train_svr(X_train, y_train,
              kernel="rbf",
              C=10.0,
              epsilon=0.01,
              gamma="scale",
              degree=3):
    model = SVR(
        kernel=kernel,
        C=C,
        epsilon=epsilon,
        gamma=gamma,
        degree=degree
    )
    model.fit(X_train, y_train)
    return model


def predict_svr(model, X):
    return model.predict(X)
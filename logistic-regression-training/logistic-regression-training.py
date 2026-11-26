
import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:

    m, n = X.shape

    w = np.zeros(n)
    b = 0.0

    for _ in range(steps):

        # Linear combination
        z = np.dot(X, w) + b

        # Sigmoid activation
        y_pred = _sigmoid(z)

        # Calculate gradients
        dw = (1/m) * np.dot(X.T, (y_pred - y))
        db = (1/m) * np.sum(y_pred - y)

        # Update weights and bias
        w = w - lr * dw
        b = b - lr * db

    return w, b

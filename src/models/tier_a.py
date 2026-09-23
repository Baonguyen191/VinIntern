import numpy as np
from sklearn.linear_model import Ridge


class TierAModel:
    def __init__(self, alpha: float = 1.0):
        self.model = Ridge(alpha=alpha, fit_intercept=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TierAModel":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

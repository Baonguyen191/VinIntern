import numpy as np
from sklearn.linear_model import Ridge


class TierBModel:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.models: dict[str, Ridge] = {}

    def fit(self, X: np.ndarray, y_dict: dict[str, np.ndarray]) -> "TierBModel":
        for name, y in y_dict.items():
            m = Ridge(alpha=self.alpha, fit_intercept=False)
            m.fit(X, y)
            self.models[name] = m
        return self

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        return {name: m.predict(X) for name, m in self.models.items()}

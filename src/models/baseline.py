import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from scipy.optimize import curve_fit
import lightgbm as lgb


class OLSModel:
    name = "ols"

    def __init__(self):
        self._model = LinearRegression()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    @property
    def coefficients(self) -> dict:
        return {
            "intercept": float(self._model.intercept_),
            "coefs": self._model.coef_.tolist(),
        }


class RidgeModel:
    name = "ridge"

    def __init__(self, alpha: float = 1.0):
        self._model = Ridge(alpha=alpha)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    @property
    def coefficients(self) -> dict:
        return {
            "intercept": float(self._model.intercept_),
            "coefs": self._model.coef_.tolist(),
        }


class LGBMModel:
    name = "lgbm"

    def __init__(self):
        self._model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=20,
            verbose=-1,
        )
        self._cat_indices: list[int] = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        categorical_indices: list[int] | None = None,
    ) -> None:
        self._cat_indices = categorical_indices or []
        self._model.fit(
            X,
            y,
            categorical_feature=self._cat_indices if self._cat_indices else "auto",
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    @property
    def coefficients(self) -> dict:
        return {"feature_importances": self._model.feature_importances_.tolist()}


def _cp5_func(temp, e_base, beta_h, beta_c, t_h, t_c):
    return e_base + beta_h * np.maximum(0, t_h - temp) + beta_c * np.maximum(
        0, temp - t_c
    )


class ChangePoint5PModel:
    name = "5p"

    def __init__(self):
        self._cp_params: np.ndarray | None = None
        self._residual_model = LinearRegression()
        self._has_residual_features = False

    def fit(
        self,
        temp: np.ndarray,
        y: np.ndarray,
        residual_X: np.ndarray | None = None,
    ) -> None:
        t_min, t_max = float(temp.min()), float(temp.max())
        t_mid = (t_min + t_max) / 2
        p0 = [np.mean(y), 0.5, 0.5, t_mid - 5, t_mid + 5]
        bounds = (
            [0, 0, 0, t_min, t_min],
            [np.max(y) * 2, np.inf, np.inf, t_max, t_max],
        )
        try:
            popt, _ = curve_fit(_cp5_func, temp, y, p0=p0, bounds=bounds, maxfev=5000)
            if popt[3] > popt[4]:
                popt[3], popt[4] = popt[4], popt[3]
                popt[1], popt[2] = popt[2], popt[1]
            self._cp_params = popt
        except RuntimeError:
            self._cp_params = np.array(p0)

        if residual_X is not None and residual_X.shape[1] > 0:
            cp_pred = _cp5_func(temp, *self._cp_params)
            residuals = y - cp_pred
            self._residual_model.fit(residual_X, residuals)
            self._has_residual_features = True
        else:
            self._has_residual_features = False

    def predict(
        self, temp: np.ndarray, residual_X: np.ndarray | None = None
    ) -> np.ndarray:
        y_pred = _cp5_func(temp, *self._cp_params)
        if self._has_residual_features and residual_X is not None:
            y_pred = y_pred + self._residual_model.predict(residual_X)
        return y_pred

    @property
    def coefficients(self) -> dict:
        names = ["e_base", "beta_h", "beta_c", "t_h", "t_c"]
        return {n: float(v) for n, v in zip(names, self._cp_params)}


def iterative_clean(
    fit_fn,
    predict_fn,
    X,
    y: np.ndarray,
    eta: float = 3.0,
    max_iter: int = 10,
) -> np.ndarray:
    mask = np.ones(len(y), dtype=bool)

    for _ in range(max_iter):
        fit_fn(X, y, mask)
        y_pred = predict_fn(X)
        residuals = y - y_pred
        mu = np.mean(residuals[mask])
        sigma = np.std(residuals[mask])
        new_mask = (residuals >= mu - eta * sigma) & (residuals <= mu + eta * sigma)
        if int(np.sum(mask) - np.sum(new_mask)) == 0:
            break
        mask = new_mask

    return mask

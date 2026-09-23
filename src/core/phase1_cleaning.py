import numpy as np

from ..models.tier_a import TierAModel
from ..models.tier_b import TierBModel


def phase1_clean_tier_a(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
    eta: float = 3.0,
    max_iter: int = 10,
) -> tuple[TierAModel, np.ndarray]:
    mask = np.ones(len(y), dtype=bool)
    model = TierAModel(alpha=alpha)

    for iteration in range(max_iter):
        model = TierAModel(alpha=alpha)
        model.fit(X[mask], y[mask])
        y_pred = model.predict(X)
        residuals = y - y_pred

        mu = np.mean(residuals[mask])
        sigma = np.std(residuals[mask])
        new_mask = (residuals >= mu - eta * sigma) & (residuals <= mu + eta * sigma)

        removed = int(np.sum(mask) - np.sum(new_mask))
        if removed == 0:
            break
        mask = new_mask

    return model, mask


def phase1_clean_tier_b(
    X: np.ndarray,
    y_dict: dict[str, np.ndarray],
    alpha: float = 1.0,
    eta: float = 3.0,
    max_iter: int = 10,
) -> tuple[TierBModel, dict[str, np.ndarray]]:
    masks = {}
    cleaned_y = {}

    for stat_name, y in y_dict.items():
        mask = np.ones(len(y), dtype=bool)
        for _ in range(max_iter):
            from sklearn.linear_model import Ridge

            m = Ridge(alpha=alpha, fit_intercept=False)
            m.fit(X[mask], y[mask])
            y_pred = m.predict(X)
            residuals = y - y_pred

            mu = np.mean(residuals[mask])
            sigma = np.std(residuals[mask])
            new_mask = (residuals >= mu - eta * sigma) & (
                residuals <= mu + eta * sigma
            )

            if int(np.sum(mask) - np.sum(new_mask)) == 0:
                break
            mask = new_mask

        masks[stat_name] = mask
        cleaned_y[stat_name] = y

    joint_mask = np.ones(len(X), dtype=bool)
    for m in masks.values():
        joint_mask &= m

    model = TierBModel(alpha=alpha)
    model.fit(X[joint_mask], {k: v[joint_mask] for k, v in y_dict.items()})

    return model, masks

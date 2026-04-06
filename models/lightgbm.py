"""LightGBM model with lag features via mlforecast."""

from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, RollingStd
from lightgbm import LGBMRegressor

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from config import LGBM_PARAMS
from models import ModelSpec


def build(
    freq: str,
    season_length: int,
    seed: int,
    lags: list[int] | None = None,
) -> ModelSpec:
    """Build an MLForecast object with LightGBM + lag features.

    Parameters
    ----------
    freq : str
        Pandas frequency string.
    season_length : int
        Seasonal period — used to derive default lags.
    seed : int
        Random seed for reproducibility.
    lags : list[int] or None
        Explicit lag values. If None, auto-generated from season_length.

    Returns
    -------
    ModelSpec
        Wraps an MLForecast object (not yet fitted).
    """
    if lags is None:
        # Generate lags: [1, 2, ..., season_length, 2*season_length]
        lags = list(range(1, season_length + 1))
        if 2 * season_length not in lags:
            lags.append(2 * season_length)

    # Lag transforms: rolling statistics at seasonal windows
    lag_transforms = {
        season_length: [
            RollingMean(window_size=season_length),
            RollingStd(window_size=season_length),
        ],
    }

    # Date features depend on frequency
    date_features = []
    if freq in ("D", "d"):
        date_features = ["dayofweek", "month"]
    elif freq in ("h", "H"):
        date_features = ["hour", "dayofweek"]
    elif freq in ("ME", "MS", "M"):
        date_features = ["month"]

    model = LGBMRegressor(
        n_estimators=LGBM_PARAMS["n_estimators"],
        learning_rate=LGBM_PARAMS["learning_rate"],
        num_leaves=LGBM_PARAMS["num_leaves"],
        random_state=seed,
        verbosity=-1,
    )

    mlf = MLForecast(
        models={"LightGBM": model},
        freq=freq,
        lags=lags,
        lag_transforms=lag_transforms,
        date_features=date_features if date_features else None,
    )
    return ModelSpec(
        name="LightGBM",
        model_type="ml",
        forecaster=mlf,
        needs_seed=True,
    )

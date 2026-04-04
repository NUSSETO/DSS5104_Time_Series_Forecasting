"""
Baseline models: Seasonal Naive and AutoARIMA via statsforecast.

These serve as the classical baselines against which deep learning
models are compared.
"""

from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive, AutoARIMA


def get_baseline_forecaster(season_length: int, freq: str) -> StatsForecast:
    """Build a StatsForecast object with Seasonal Naive + AutoARIMA.

    Parameters
    ----------
    season_length : int
        Seasonal period (e.g., 12 for monthly, 7 for daily, 24 for hourly).
    freq : str
        Pandas frequency string (e.g., 'ME', 'D', 'h').

    Returns
    -------
    StatsForecast
        Fitted forecaster with both baseline models.
    """
    models = [
        SeasonalNaive(season_length=season_length),
        AutoARIMA(season_length=season_length),
    ]
    sf = StatsForecast(
        models=models,
        freq=freq,
        n_jobs=-1,  # use all CPUs
    )
    return sf

"""Model definitions for time-series forecasting.

All model factory functions return a ModelSpec, enabling uniform handling
in the walk-forward evaluation engine regardless of underlying framework.
"""

from dataclasses import dataclass
from typing import Any, Literal

ModelType = Literal["stats", "ml", "neural"]


@dataclass
class ModelSpec:
    """Uniform wrapper returned by every model factory function.

    Attributes
    ----------
    name : str
        Model name as it appears in prediction column headers and result CSVs
        (e.g. "PatchTST", "SeasonalNaive", "LightGBM").
    model_type : {"stats", "ml", "neural"}
        Framework type — determines which fit/predict API to call.
        - "stats"  : StatsForecast  — sf.fit(df) / sf.predict(h=horizon)
        - "ml"     : MLForecast     — mlf.fit(df) / mlf.predict(h=horizon)
        - "neural" : NeuralForecast — nf.fit(df=df, val_size=h) / nf.predict()
    forecaster : Any
        The underlying StatsForecast | MLForecast | NeuralForecast object.
    needs_seed : bool
        False for deterministic models (SeasonalNaive, AutoARIMA); True for
        ML/DL models where results vary across seeds.
    """

    name: str
    model_type: ModelType
    forecaster: Any
    needs_seed: bool

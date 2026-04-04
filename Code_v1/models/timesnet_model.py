"""
TimesNet model via neuralforecast.

Converts 1D time series into 2D tensors via FFT to capture multi-periodicity,
then applies 2D CNNs. Chosen for architectural diversity — adding a CNN-based
approach alongside Transformer (PatchTST), MLP (TiDE, DLinear),
deep residual (N-BEATS), and RNN (DeepAR).
"""

from neuralforecast import NeuralForecast
from neuralforecast.models import TimesNet

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from config import (
    MAX_STEPS, BATCH_SIZE, LEARNING_RATE, EARLY_STOP_PATIENCE,
    VAL_CHECK_STEPS, TIMESNET_PARAMS,
)


def get_timesnet(
    horizon: int,
    input_size: int,
    freq: str,
    seed: int,
    max_steps: int | None = None,
) -> NeuralForecast:
    """Build NeuralForecast with TimesNet.

    Parameters
    ----------
    horizon : int
        Forecast horizon (h).
    input_size : int
        Lookback window length.
    freq : str
        Pandas frequency string.
    seed : int
        Random seed for reproducibility.
    max_steps : int or None
        Override default MAX_STEPS (useful for smoke tests).

    Returns
    -------
    NeuralForecast
        Configured with TimesNet (not yet fitted).
    """
    steps = max_steps if max_steps is not None else MAX_STEPS

    model = TimesNet(
        h=horizon,
        input_size=input_size,
        top_k=TIMESNET_PARAMS["top_k"],
        num_kernels=TIMESNET_PARAMS["num_kernels"],
        hidden_size=TIMESNET_PARAMS["d_model"],
        conv_hidden_size=TIMESNET_PARAMS["d_ff"],
        encoder_layers=TIMESNET_PARAMS["e_layers"],
        max_steps=steps,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        random_seed=seed,
        early_stop_patience_steps=EARLY_STOP_PATIENCE,
        val_check_steps=VAL_CHECK_STEPS,
        scaler_type="standard",
    )

    nf = NeuralForecast(models=[model], freq=freq)
    return nf

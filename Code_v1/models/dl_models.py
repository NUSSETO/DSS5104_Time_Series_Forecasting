"""
Core deep learning models via neuralforecast:
  - PatchTST  (Transformer with patches)
  - N-BEATS   (backward/forward residual links)
  - TiDE      (MLP encoder-decoder)
  - DeepAR    (RNN autoregressive, probabilistic)
  - DLinear   (single linear layer — critical baseline)
  - TimesNet  (2D CNN via FFT — CNN architectural diversity)
"""

from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST, NBEATS, TiDE, DeepAR, DLinear, TimesNet

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from config import (
    MAX_STEPS, BATCH_SIZE, LEARNING_RATE, EARLY_STOP_PATIENCE,
    VAL_CHECK_STEPS, PATCHTST_PARAMS, NBEATS_PARAMS, TIDE_PARAMS, DEEPAR_PARAMS,
    TIMESNET_PARAMS,
)


def get_dl_models(
    horizon: int,
    input_size: int,
    freq: str,
    seed: int,
    max_steps: int | None = None,
) -> NeuralForecast:
    """Build NeuralForecast with PatchTST, N-BEATS, TiDE, DeepAR, DLinear, TimesNet.

    All 6 models share one NeuralForecast object so neuralforecast's internal
    data preprocessing (scaling, windowing, DataLoader construction) runs only
    once per fit call instead of once per model group.

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
        Configured with 6 models (not yet fitted).
    """
    steps = max_steps if max_steps is not None else MAX_STEPS

    models = [
        PatchTST(
            h=horizon,
            input_size=input_size,
            patch_len=PATCHTST_PARAMS["patch_len"],
            stride=PATCHTST_PARAMS["stride"],
            n_heads=PATCHTST_PARAMS["n_heads"],
            hidden_size=PATCHTST_PARAMS["hidden_size"],
            encoder_layers=PATCHTST_PARAMS["encoder_layers"],
            max_steps=steps,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            random_seed=seed,
            early_stop_patience_steps=EARLY_STOP_PATIENCE,
            val_check_steps=VAL_CHECK_STEPS,
            scaler_type="standard",
        ),
        NBEATS(
            h=horizon,
            input_size=input_size,
            stack_types=NBEATS_PARAMS["stack_types"],
            n_blocks=NBEATS_PARAMS["n_blocks"],
            mlp_units=NBEATS_PARAMS["mlp_units"],
            max_steps=steps,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            random_seed=seed,
            early_stop_patience_steps=EARLY_STOP_PATIENCE,
            val_check_steps=VAL_CHECK_STEPS,
            scaler_type="standard",
        ),
        TiDE(
            h=horizon,
            input_size=input_size,
            hidden_size=TIDE_PARAMS["hidden_size"],
            decoder_output_dim=TIDE_PARAMS["decoder_output_dim"],
            num_encoder_layers=TIDE_PARAMS["num_encoder_layers"],
            num_decoder_layers=TIDE_PARAMS["num_decoder_layers"],
            max_steps=steps,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            random_seed=seed,
            early_stop_patience_steps=EARLY_STOP_PATIENCE,
            val_check_steps=VAL_CHECK_STEPS,
            scaler_type="standard",
        ),
        DeepAR(
            h=horizon,
            input_size=input_size,
            lstm_hidden_size=DEEPAR_PARAMS["hidden_size"],
            lstm_n_layers=DEEPAR_PARAMS["n_layers"],
            max_steps=steps,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            random_seed=seed,
            early_stop_patience_steps=EARLY_STOP_PATIENCE,
            val_check_steps=VAL_CHECK_STEPS,
            scaler_type="standard",
        ),
        DLinear(
            h=horizon,
            input_size=input_size,
            max_steps=steps,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            random_seed=seed,
            early_stop_patience_steps=EARLY_STOP_PATIENCE,
            val_check_steps=VAL_CHECK_STEPS,
            scaler_type="standard",
        ),
        TimesNet(
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
        ),
    ]

    nf = NeuralForecast(models=models, freq=freq)
    return nf

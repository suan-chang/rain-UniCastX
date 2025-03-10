import logging
import torch

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

_logger = logging.getLogger(__name__)

ACTIVATIONS = [
    "ReLU",
    "Softplus",
    "Tanh",
    "SELU",
    "LeakyReLU",
    "PReLU",
    "Sigmoid",
    "GELU",
]
OFFSET_ALIAS_MAP = {
    "W": "W",
    "d": "D",
    "D": "D",
    "days": "D",
    "day": "D",
    "hours": "H",
    "hour": "H",
    "hr": "H",
    "h": "H",
    "H": "H",
    "m": "min",
    "minute": "min",
    "min": "min",
    "minutes": "min",
    "T": "min",
    "M": "MS",
    "MS": "MS",
    "months": "MS",
    "month": "MS",
}
MODEL_PARAMS_DICT = {
    "min": {"input_size": 512, "model_horizon": 128},
    "H": {"input_size": 512, "model_horizon": 128},
    "D": {"input_size": 512, "model_horizon": 60},
    "W": {"input_size": 512, "model_horizon": 60},
    "MS": {"input_size": 512, "model_horizon": 24},
}
SEP_TIME_COL_NAMES = [
    "sin_time",
    "cos_time",
    "sin_week",
    "cos_week",
    "sin_month",
    "cos_month",
]


def check_params(
    freq: str,
    horizon: int,
    level: Optional[float] = None,
) -> Tuple[str, int, Optional[float]]:
    """
    Check input parameters.

    Args:
        freq: The input frequency.
        horizon: The predict length.
        level: Prediction interval.

    Returns:
        Parsed parameters.
    """
    if level:
        assert (
            level >= 0 and level <= 1
        ), f"level must be greater than or equal to 0 and less than or equal to 1, but got {level}."

    assert (
        freq in OFFSET_ALIAS_MAP
    ), f"freq must be the following values [{', '.join(OFFSET_ALIAS_MAP.keys())}], but got {freq}."
    freq = OFFSET_ALIAS_MAP[freq]

    horizon = min(horizon, MODEL_PARAMS_DICT[freq]["model_horizon"])
    assert horizon > 0, f"horizon must be greater than 0, but got {horizon}."

    return freq, horizon, level


def data_preprocess(
    data: pd.DataFrame,
    time_col: str,
    target_col: str,
    freq: str,
    horizon: int,
    min_input_size: int,
    max_exog_nums: int,
    torch_type: torch.dtype,
    device: str,
    exog_cols: Optional[List[str]] = None,
    level: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert a dataframe to tensor.

    Args:
        data: The raw dataframe.
        time_col: Time column.
        target_col: Target column.
        freq: The input frequency.
        horizon: The prediction length.
        min_input_size: The minimum input length of the time series model.
        max_exog_nums: The maximum number of exogenous variables.
        torch_type: Torch type.
        device: Device.
        exog_cols: List of exogenous variables or None.
        level: Prediction interval.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    freq, horizon, level = check_params(freq=freq, horizon=horizon, level=level)
    assert (
        len(data) >= min_input_size
    ), f"The length of time_col must be greater than or equal to {min_input_size}, but got {len(data)}."

    columns = [time_col, target_col]
    if exog_cols:
        if len(exog_cols) > max_exog_nums:
            _logger.warning(
                f"The maximum number of supported exogenous columns is {max_exog_nums}, "
                f"but got {len(exog_cols)}. The first {max_exog_nums} columns will be used by default."
            )
            exog_cols = exog_cols[:max_exog_nums]

        columns += exog_cols

    data = data[columns]

    max_input_size = (
        3 * MODEL_PARAMS_DICT[freq]["input_size"] + horizon
        if level
        else MODEL_PARAMS_DICT[freq]["input_size"]
    )

    if len(data) > max_input_size:
        data = data.tail(max_input_size).reset_index(drop=True)
        _logger.warning(
            f"The length of inputs is greater than the maximum number, only the tail {max_input_size} points will be used."
        )

    data[time_col] = pd.to_datetime(data[time_col], errors="coerce")
    if "MS" in freq or "W" in freq:
        month_of_year = data[time_col].dt.month
        data["sin_month"] = np.sin(
            2 * np.pi * month_of_year / (month_of_year.max() + 0.0001)
        )
        data["cos_month"] = np.cos(
            2 * np.pi * month_of_year / (month_of_year.max() + 0.0001)
        )
        data["sin_time"] = 0
        data["cos_time"] = 0
        data["sin_week"] = 0
        data["cos_week"] = 0
    else:
        month_of_year = data[time_col].dt.month
        data["sin_month"] = np.sin(
            2 * np.pi * month_of_year / (month_of_year.max() + 0.0001)
        )
        data["cos_month"] = np.cos(
            2 * np.pi * month_of_year / (month_of_year.max() + 0.0001)
        )
        time_of_day = (
            data[time_col].dt.hour * 3600
            + data[time_col].dt.minute * 60
            + data[time_col].dt.second
        )
        data["sin_time"] = np.sin(
            2 * np.pi * time_of_day / (time_of_day.max() + 0.0001)
        )
        data["cos_time"] = np.cos(
            2 * np.pi * time_of_day / (time_of_day.max() + 0.0001)
        )
        day_of_week = data[time_col].dt.weekday + 1
        data["sin_week"] = np.sin(
            2 * np.pi * day_of_week / (day_of_week.max() + 0.0001)
        )
        data["cos_week"] = np.cos(
            2 * np.pi * day_of_week / (day_of_week.max() + 0.0001)
        )

    def _value_cols_to_numpy(df: pd.DataFrame, value_cols: list) -> np.ndarray:
        data = df[value_cols].to_numpy()
        if data.dtype not in (np.float32, np.float64):
            data = data.astype(np.float32)
        return data

    time_col_tensor = torch.Tensor(data[SEP_TIME_COL_NAMES].to_numpy()).to(torch_type)
    target_col_tensor = torch.Tensor(_value_cols_to_numpy(data, [target_col])).to(
        torch_type
    )
    exog_cols = [
        col
        for col in data.columns
        if col not in SEP_TIME_COL_NAMES + [time_col, target_col]
    ]
    if len(exog_cols) == 0:
        exog_cols_tensor = torch.zeros(
            (data[target_col].shape[0], max_exog_nums),
            dtype=torch_type,
        )
    else:
        current_exog = torch.Tensor(_value_cols_to_numpy(data, exog_cols)).to(
            torch_type
        )
        if current_exog.shape[1] < max_exog_nums:
            padding = torch.zeros(
                (
                    data[target_col].shape[0],
                    max_exog_nums - current_exog.shape[1],
                ),
                dtype=torch_type,
            )
            exog_cols_tensor = torch.cat([current_exog, padding], dim=1)
        elif current_exog.shape[1] >= max_exog_nums:
            exog_cols_tensor = current_exog[:, :max_exog_nums]

    insample_time = torch.unsqueeze(time_col_tensor, dim=0).to(device)
    insample_y = torch.unsqueeze(target_col_tensor, dim=0).to(device)
    insample_exog = torch.unsqueeze(exog_cols_tensor, dim=0).to(device)

    return insample_time, insample_y, insample_exog

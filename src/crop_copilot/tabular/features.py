from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = {
    "date",
    "market",
    "crop",
    "modal_price",
    "min_price",
    "max_price",
    "rainfall_mm",
    "temp_c",
    "humidity",
}

CATEGORICAL_COLUMNS = ["market", "crop"]
NUMERIC_FEATURES = [
    "min_price",
    "max_price",
    "rainfall_mm",
    "temp_c",
    "humidity",
    "day_of_year",
    "month",
    "lag_1",
    "lag_7",
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_mean_14",
]
FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_FEATURES
TARGET_COLUMN = "target_modal_price_7d"
FORECAST_HORIZON_DAYS = 7


def validate_market_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Market data is missing required columns: {sorted(missing)}")
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    numeric = REQUIRED_COLUMNS - {"date", "market", "crop"}
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="raise")
    if result.duplicated(["date", "market", "crop"]).any():
        raise ValueError("Duplicate date + market + crop rows found")
    if (result[["modal_price", "min_price", "max_price"]] < 0).any().any():
        raise ValueError("Prices must be non-negative")
    return result.sort_values(["market", "crop", "date"]).reset_index(drop=True)


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = validate_market_frame(frame)
    result["day_of_year"] = result["date"].dt.dayofyear
    result["month"] = result["date"].dt.month
    groups = result.groupby(["market", "crop"], observed=True)["modal_price"]
    result["lag_1"] = groups.shift(1)
    result["lag_7"] = groups.shift(7)
    result["rolling_mean_7"] = groups.transform(
        lambda values: values.shift(1).rolling(7, min_periods=3).mean()
    )
    result["rolling_std_7"] = groups.transform(
        lambda values: values.shift(1).rolling(7, min_periods=3).std()
    )
    result["rolling_mean_14"] = groups.transform(
        lambda values: values.shift(1).rolling(14, min_periods=7).mean()
    )
    result[TARGET_COLUMN] = groups.shift(-FORECAST_HORIZON_DAYS)
    return result.dropna(subset=NUMERIC_FEATURES).reset_index(drop=True)

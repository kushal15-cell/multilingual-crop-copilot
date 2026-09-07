from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from crop_copilot.schemas import MarketContext
from crop_copilot.tabular.features import FEATURE_COLUMNS, build_features


class PriceForecaster:
    def __init__(self, model_path: Path, data_path: Path) -> None:
        self.model_path = model_path
        self.data_path = data_path
        self.bundle = joblib.load(model_path) if model_path.exists() else None

    def context(self, crop: str, market: str | None) -> MarketContext:
        if not self.data_path.exists():
            return MarketContext(crop=crop, market=market, unavailable_reason="Market CSV missing")
        raw = pd.read_csv(self.data_path)
        selection = raw[raw["crop"].str.casefold() == crop.casefold()].copy()
        if market:
            selection = selection[selection["market"].str.casefold() == market.casefold()]
        if selection.empty:
            return MarketContext(crop=crop, market=market, unavailable_reason="No matching history")
        selection["date"] = pd.to_datetime(selection["date"])
        latest = selection.sort_values("date").iloc[-1]
        source = str(latest.get("data_source", "unknown"))
        predicted = None
        trend = "unavailable"
        if self.bundle is not None:
            featured = build_features(selection)
            if not featured.empty:
                last_features = featured.iloc[[-1]][FEATURE_COLUMNS]
                predicted = float(self.bundle["pipeline"].predict(last_features)[0])
                change = (predicted - float(latest["modal_price"])) / max(
                    float(latest["modal_price"]), 1
                )
                trend = "rising" if change > 0.03 else "falling" if change < -0.03 else "stable"
        return MarketContext(
            crop=crop,
            market=str(latest["market"]),
            latest_modal_price=float(latest["modal_price"]),
            predicted_7d_price=predicted,
            trend=trend,
            as_of=latest["date"].to_pydatetime(),
            source=source,
            is_synthetic=source.casefold().startswith("synthetic"),
        )

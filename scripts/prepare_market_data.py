from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from crop_copilot.tabular.features import validate_market_frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map a provider CSV to the project's normalized mandi data contract"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--date-column", default="date")
    parser.add_argument("--market-column", default="market")
    parser.add_argument("--crop-column", default="crop")
    parser.add_argument("--modal-price-column", default="modal_price")
    parser.add_argument("--min-price-column", default="min_price")
    parser.add_argument("--max-price-column", default="max_price")
    parser.add_argument("--rainfall-column", default="rainfall_mm")
    parser.add_argument("--temperature-column", default="temp_c")
    parser.add_argument("--humidity-column", default="humidity")
    parser.add_argument(
        "--price-multiplier",
        type=float,
        default=1.0,
        help="Convert the provider unit to INR/quintal, e.g. multiply INR/kg by 100",
    )
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    mapping = {
        args.date_column: "date",
        args.market_column: "market",
        args.crop_column: "crop",
        args.modal_price_column: "modal_price",
        args.min_price_column: "min_price",
        args.max_price_column: "max_price",
        args.rainfall_column: "rainfall_mm",
        args.temperature_column: "temp_c",
        args.humidity_column: "humidity",
    }
    normalized = frame.rename(columns=mapping)[list(mapping.values())]
    normalized["market"] = normalized["market"].astype(str).str.strip()
    normalized["crop"] = normalized["crop"].astype(str).str.strip().str.casefold()
    for column in ["modal_price", "min_price", "max_price"]:
        normalized[column] = pd.to_numeric(normalized[column]) * args.price_multiplier
    normalized = validate_market_frame(normalized)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(args.output, index=False)
    print(f"Wrote {len(normalized)} normalized rows to {args.output}")


if __name__ == "__main__":
    main()


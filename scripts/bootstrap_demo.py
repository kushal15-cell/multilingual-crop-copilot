from __future__ import annotations

import math
import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd


def main() -> None:
    random.seed(42)
    output = Path("data/processed/market_prices.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    start = date.today() - timedelta(days=220)
    markets = [
        ("Davangere", 0),
        ("Bengaluru", 180),
        ("Mysuru", 110),
        ("Mangaluru", 145),
        ("Hubballi", 35),
        ("Belagavi", 60),
        ("Shivamogga", 25),
    ]
    for market, offset in markets:
        for crop, base in [("tomato", 1800), ("potato", 1400)]:
            price = base + offset
            for day in range(220):
                current = start + timedelta(days=day)
                seasonal = 180 * math.sin(day / 22)
                price = max(
                    300,
                    0.8 * price + 0.2 * (base + offset + seasonal) + random.gauss(0, 35),
                )
                rows.append(
                    {
                        "date": current.isoformat(),
                        "market": market,
                        "crop": crop,
                        "modal_price": round(price, 2),
                        "min_price": round(price * 0.88, 2),
                        "max_price": round(price * 1.12, 2),
                        "rainfall_mm": max(0, round(random.gauss(3, 5), 2)),
                        "temp_c": round(27 + 4 * math.sin(day / 45) + random.gauss(0, 1), 2),
                        "humidity": round(min(95, max(30, 65 + random.gauss(0, 8))), 2),
                        "data_source": "synthetic_demo_v2",
                    }
                )
    pd.DataFrame(rows).to_csv(output, index=False)
    Path("artifacts/cv").mkdir(parents=True, exist_ok=True)
    Path("artifacts/price").mkdir(parents=True, exist_ok=True)
    print(f"Wrote {len(rows)} synthetic demonstration rows to {output}")
    print("These values are synthetic and must never be presented as real mandi data.")


if __name__ == "__main__":
    main()

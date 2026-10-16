"""Generate a reproducible synthetic daily-demand dataset."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SEED = 20261016

def main():
    rng = np.random.default_rng(SEED)
    dates = pd.date_range("2024-01-01", periods=730, freq="D")
    rows, products = [], []
    for i in range(1, 25):
        sku = f"SKU-{i:03d}"
        base = float(rng.integers(5, 42))
        unit_cost = round(float(rng.uniform(2.5, 45)), 2)
        price = round(unit_cost * float(rng.uniform(1.35, 2.2)), 2)
        lead_min = int(rng.integers(2, 5))
        lead_max = int(lead_min + rng.integers(2, 7))
        products.append({
            "sku": sku, "base_daily_demand": base, "unit_cost": unit_cost,
            "selling_price": price, "lead_time_min_days": lead_min,
            "lead_time_max_days": lead_max
        })
        weekday_factor = np.where(dates.dayofweek >= 5, 0.72, 1.0)
        annual_factor = 1 + 0.22 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
        promo = rng.random(len(dates)) < 0.035
        promo_factor = np.where(promo, rng.uniform(1.4, 2.3, len(dates)), 1.0)
        lam = base * weekday_factor * annual_factor * promo_factor
        demand = rng.poisson(lam).astype(int)
        for date, units, is_promo in zip(dates, demand, promo):
            rows.append({"date": date.date().isoformat(), "sku": sku,
                         "demand_units": int(units), "promo_day": bool(is_promo)})
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(DATA_DIR / "daily_demand.csv", index=False)
    pd.DataFrame(products).to_csv(DATA_DIR / "products.csv", index=False)
    print(f"Wrote {len(rows):,} daily demand rows and {len(products)} product records.")

if __name__ == "__main__":
    main()

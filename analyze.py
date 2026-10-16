"""Compare inventory policies against a shared synthetic demand history."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
POLICIES = {
    "baseline": 1.0,
    "higher_safety_stock": 1.8,
    "lean_inventory": 0.35,
}
ANNUAL_HOLDING_RATE = 0.24
LOST_SALES_MARGIN_RATE = 0.45
SEED = 20261017

def simulate_policy(demand: np.ndarray, lead_time: int, unit_cost: float,
                    unit_price: float, safety_multiplier: float, rng) -> dict:
    """Daily-review (s,S)-style simulation with lost sales and fixed order quantity.

    Reorder point = mean demand during lead time + safety buffer.
    When inventory position reaches reorder point, order enough to reach target stock.
    Orders arrive after a sampled lead time. A simplified fixed lead time is used per SKU
    simulation for reproducibility and ease of interpretation.
    """
    n = len(demand)
    mean_d = float(np.mean(demand))
    std_d = float(np.std(demand, ddof=1))
    safety = safety_multiplier * 1.65 * std_d * np.sqrt(lead_time)
    reorder_point = max(1, int(np.ceil(mean_d * lead_time + safety)))
    target_stock = max(reorder_point + 1, int(np.ceil(mean_d * (lead_time + 7) + safety)))
    on_hand = target_stock
    pipeline = []  # (arrival day, units)
    lost_units = 0
    total_demand = int(demand.sum())
    inventory_sum = 0
    order_count = 0
    daily_rows = []
    for day, requested in enumerate(demand):
        arriving = [order for order in pipeline if order[0] <= day]
        if arriving:
            on_hand += sum(order[1] for order in arriving)
            pipeline = [order for order in pipeline if order[0] > day]
        fulfilled = min(on_hand, int(requested))
        lost = int(requested) - fulfilled
        on_hand -= fulfilled
        lost_units += lost
        inventory_sum += on_hand
        daily_rows.append((day, int(requested), fulfilled, lost, on_hand))
        inventory_position = on_hand + sum(q for _, q in pipeline)
        if inventory_position <= reorder_point:
            order_qty = max(0, target_stock - inventory_position)
            if order_qty:
                # Use fixed SKU-level lead time to keep policy comparison interpretable.
                pipeline.append((day + lead_time, int(order_qty)))
                order_count += 1
    avg_inventory = inventory_sum / n
    holding_cost = avg_inventory * unit_cost * ANNUAL_HOLDING_RATE * (n / 365.25)
    lost_sales_cost = lost_units * unit_price * LOST_SALES_MARGIN_RATE
    service_level = 1.0 if total_demand == 0 else (total_demand - lost_units) / total_demand
    return {
        "service_level": service_level,
        "total_demand_units": total_demand,
        "fulfilled_units": total_demand - lost_units,
        "lost_units": int(lost_units),
        "average_on_hand_units": avg_inventory,
        "holding_cost_proxy": holding_cost,
        "estimated_lost_sales_cost": lost_sales_cost,
        "combined_cost_proxy": holding_cost + lost_sales_cost,
        "orders_placed": order_count,
        "reorder_point_units": reorder_point,
        "target_stock_units": target_stock,
    }

def main():
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    demand_df = pd.read_csv(DATA / "daily_demand.csv", parse_dates=["date"])
    products = pd.read_csv(DATA / "products.csv")
    required = {"date", "sku", "demand_units", "promo_day"}
    if not required.issubset(demand_df.columns):
        raise ValueError(f"Missing demand columns: {required - set(demand_df.columns)}")
    if demand_df["demand_units"].isna().any() or (demand_df["demand_units"] < 0).any():
        raise ValueError("Demand must be complete and non-negative.")
    if demand_df.duplicated(["date", "sku"]).any():
        raise ValueError("Duplicate date/SKU rows found.")
    if products["sku"].duplicated().any():
        raise ValueError("Product SKU must be unique.")
    if set(demand_df["sku"]) != set(products["sku"]):
        raise ValueError("Demand SKUs and product SKUs do not match.")

    rng = np.random.default_rng(SEED)
    records = []
    for product in products.itertuples(index=False):
        sku_df = demand_df[demand_df["sku"] == product.sku].sort_values("date")
        series = sku_df["demand_units"].to_numpy(dtype=int)
        lead = int(np.ceil((product.lead_time_min_days + product.lead_time_max_days) / 2))
        for policy, multiplier in POLICIES.items():
            metrics = simulate_policy(series, lead, float(product.unit_cost),
                                      float(product.selling_price), multiplier, rng)
            records.append({"sku": product.sku, "policy": policy,
                            "unit_cost": product.unit_cost,
                            "selling_price": product.selling_price, **metrics})
    detail = pd.DataFrame(records)
    summary = detail.groupby("policy", as_index=False).agg(
        service_level=("service_level", "mean"),
        lost_units=("lost_units", "sum"),
        average_on_hand_units=("average_on_hand_units", "mean"),
        holding_cost_proxy=("holding_cost_proxy", "sum"),
        estimated_lost_sales_cost=("estimated_lost_sales_cost", "sum"),
        combined_cost_proxy=("combined_cost_proxy", "sum"),
        orders_placed=("orders_placed", "sum"),
    )
    summary["service_level"] = summary["service_level"].clip(0, 1)
    summary.to_csv(OUT / "policy_summary.csv", index=False)
    detail.to_csv(OUT / "product_policy_summary.csv", index=False)

    # Plot 1: cost trade-off
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for row in summary.itertuples(index=False):
        ax.scatter(row.average_on_hand_units, row.service_level, s=100)
        ax.annotate(row.policy.replace("_", " "), (row.average_on_hand_units, row.service_level),
                    xytext=(7, 6), textcoords="offset points", fontsize=9)
    ax.set(title="Inventory availability vs. average stock",
           xlabel="Mean on-hand units per SKU", ylabel="Mean product-level fill rate")
    ax.set_ylim(0, 1.03)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "service_vs_inventory.png", dpi=160)
    plt.close(fig)

    # Plot 2: cost composition
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(summary))
    ax.bar(x, summary["holding_cost_proxy"], label="Holding-cost proxy")
    ax.bar(x, summary["estimated_lost_sales_cost"],
           bottom=summary["holding_cost_proxy"], label="Estimated lost-sales cost")
    ax.set_xticks(x, summary["policy"].str.replace("_", " "))
    ax.set_ylabel("Cost proxy (arbitrary currency units)")
    ax.set_title("Illustrative cost trade-off by policy")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "cost_composition.png", dpi=160)
    plt.close(fig)

    # Validation of output invariants
    assert set(summary["policy"]) == set(POLICIES)
    assert summary["service_level"].between(0, 1).all()
    assert (summary["lost_units"] >= 0).all()
    assert (summary["combined_cost_proxy"] >= 0).all()
    assert len(detail) == len(products) * len(POLICIES)
    print("Validation passed.")
    print(f"Demand rows: {len(demand_df):,}; products: {len(products)}; policy-SKU rows: {len(detail)}")
    print(summary.round(3).to_string(index=False))
    print(f"Outputs written to: {OUT}")

if __name__ == "__main__":
    main()

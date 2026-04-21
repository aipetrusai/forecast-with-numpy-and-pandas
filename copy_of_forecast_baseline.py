from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backend.app.bigquery_client import get_bigquery_client
from backend.app.config import BQ_DATASET_ID, GCP_PROJECT_ID


OUTPUT_DIR = Path("model/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_campaign_period_data() -> pd.DataFrame:
    """
    Trae el histórico diario enriquecido y lo cruza con el presupuesto demo
    por campaign_id y por fecha dentro del periodo presupuestario.
    """
    client = get_bigquery_client()

    query = f"""
    SELECT
      e.account,
      e.campaign_id,
      e.date,
      e.cost,
      e.cumulative_cost,
      e.avg_cost_7d,
      b.budget_name,
      b.period_start,
      b.period_end,
      b.budget_amount,
      b.projection_days,
      b.budget_source,
      b.is_estimated
    FROM `{GCP_PROJECT_ID}.{BQ_DATASET_ID}.vw_campaign_daily_enriched` e
    JOIN `{GCP_PROJECT_ID}.{BQ_DATASET_ID}.tb_campaign_budget_demo` b
      ON e.campaign_id = b.campaign_id
     AND e.date BETWEEN b.period_start AND b.period_end
    ORDER BY e.campaign_id, b.budget_name, e.date
    """

    rows = [dict(row.items()) for row in client.query(query).result()]
    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("La consulta a BigQuery no devolvió filas.")

    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    data["date"] = pd.to_datetime(data["date"])
    data["period_start"] = pd.to_datetime(data["period_start"])
    data["period_end"] = pd.to_datetime(data["period_end"])

    numeric_cols = [
        "cost",
        "cumulative_cost",
        "avg_cost_7d",
        "budget_amount",
        "projection_days",
    ]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.sort_values(["campaign_id", "budget_name", "date"]).reset_index(drop=True)
    data["weekday"] = data["date"].dt.dayofweek

    data["avg_cost_28d"] = (
        data.groupby(["campaign_id", "budget_name"])["cost"]
        .transform(lambda s: s.rolling(window=28, min_periods=1).mean())
    )

    data["days_remaining_in_period"] = (
        data["period_end"] - data["date"]
    ).dt.days.clip(lower=0)

    return data


def compute_trend_factor(avg_cost_7d: float, avg_cost_28d: float) -> float:
    if pd.isna(avg_cost_28d) or avg_cost_28d == 0:
        return 1.0

    factor = avg_cost_7d / avg_cost_28d
    return float(np.clip(factor, 0.5, 1.8))


def forecast_future_daily_values(group: pd.DataFrame, horizon_days: int) -> list[float]:
    """
    Forecast simple:
    - patrón por día de semana
    - ajustado por tendencia reciente
    """
    latest = group.iloc[-1]
    avg_cost_7d = float(latest["avg_cost_7d"])
    avg_cost_28d = float(latest["avg_cost_28d"])
    trend_factor = compute_trend_factor(avg_cost_7d, avg_cost_28d)

    weekday_means = group.groupby("weekday")["cost"].mean().to_dict()

    future_dates = pd.date_range(
        start=latest["date"] + pd.Timedelta(days=1),
        periods=horizon_days,
        freq="D",
    )

    values = []
    for d in future_dates:
        weekday = d.dayofweek
        base_value = weekday_means.get(weekday, avg_cost_7d)
        values.append(float(base_value * trend_factor))

    return values


def classify_risk(projected_cost: float, budget_amount: float) -> str:
    if pd.isna(budget_amount) or budget_amount <= 0:
        return "SIN_BUDGET"

    if projected_cost >= budget_amount:
        return "RIESGO"
    if projected_cost >= budget_amount * 0.90:
        return "VIGILAR"
    return "OK"


def estimate_days_to_overspend(
    cumulative_cost: float,
    budget_amount: float,
    future_values: list[float],
) -> int | None:
    running = cumulative_cost

    for i, value in enumerate(future_values, start=1):
        running += value
        if running >= budget_amount:
            return i

    return None


def build_forecast_table(df: pd.DataFrame) -> pd.DataFrame:
    data = prepare_data(df)

    results = []

    for (campaign_id, budget_name), group in data.groupby(["campaign_id", "budget_name"]):
        group = group.sort_values("date")
        latest = group.iloc[-1]

        forecast_7_values = forecast_future_daily_values(group, horizon_days=7)
        forecast_14_values = forecast_future_daily_values(group, horizon_days=14)

        forecast_next_7d = float(np.sum(forecast_7_values))
        forecast_next_14d = float(np.sum(forecast_14_values))

        cumulative_cost = float(latest["cumulative_cost"])
        budget_amount = float(latest["budget_amount"])
        projection_days = int(latest["projection_days"])
        days_remaining = int(latest["days_remaining_in_period"])

        effective_projection_days = min(projection_days, max(days_remaining, 0))

        future_values_for_period = forecast_future_daily_values(
            group, horizon_days=effective_projection_days
        )

        projected_cost_until_period_end = float(
            cumulative_cost + np.sum(future_values_for_period)
        )

        remaining_margin_until_period_end = float(
            budget_amount - projected_cost_until_period_end
        )

        risk_status = classify_risk(
            projected_cost=projected_cost_until_period_end,
            budget_amount=budget_amount,
        )

        days_to_overspend = estimate_days_to_overspend(
            cumulative_cost=cumulative_cost,
            budget_amount=budget_amount,
            future_values=forecast_14_values,
        )

        risk_score = float(
            np.clip((projected_cost_until_period_end / budget_amount) * 100, 0, 200)
        ) if budget_amount > 0 else np.nan

        results.append(
            {
                "account": latest["account"],
                "campaign_id": campaign_id,
                "budget_name": budget_name,
                "date": latest["date"],
                "period_start": latest["period_start"],
                "period_end": latest["period_end"],
                "cost": float(latest["cost"]),
                "cumulative_cost": cumulative_cost,
                "avg_cost_7d": float(latest["avg_cost_7d"]),
                "avg_cost_28d": float(latest["avg_cost_28d"]),
                "budget_amount": budget_amount,
                "projection_days": projection_days,
                "days_remaining_in_period": days_remaining,
                "effective_projection_days": effective_projection_days,
                "forecast_next_7d": forecast_next_7d,
                "forecast_next_14d": forecast_next_14d,
                "projected_cost_until_period_end": projected_cost_until_period_end,
                "remaining_margin_until_period_end": remaining_margin_until_period_end,
                "days_to_overspend": days_to_overspend,
                "risk_score": round(risk_score, 2),
                "risk_status": risk_status,
                "budget_source": latest["budget_source"],
                "is_estimated": latest["is_estimated"],
            }
        )

    result_df = pd.DataFrame(results)

    if not result_df.empty:
        risk_order = {"RIESGO": 0, "VIGILAR": 1, "OK": 2, "SIN_BUDGET": 3}
        result_df["risk_order"] = result_df["risk_status"].map(risk_order).fillna(99)
        result_df = result_df.sort_values(
            ["risk_order", "remaining_margin_until_period_end"],
            ascending=[True, True],
        ).drop(columns=["risk_order"])

    return result_df


def main():
    print("Consultando BigQuery...")
    raw_df = fetch_campaign_period_data()

    print("Construyendo forecast...")
    forecast_df = build_forecast_table(raw_df)

    output_path = OUTPUT_DIR / "forecast_preview.csv"
    forecast_df.to_csv(output_path, index=False)

    print("\nTop 10 campañas por riesgo:")
    print(
        forecast_df[
            [
                "campaign_id",
                "budget_name",
                "budget_amount",
                "cumulative_cost",
                "forecast_next_7d",
                "projected_cost_until_period_end",
                "remaining_margin_until_period_end",
                "days_to_overspend",
                "risk_status",
            ]
        ].head(10).to_string(index=False)
    )

    print(f"\nCSV guardado en: {output_path}")


if __name__ == "__main__":
    main()
"""Generate both submission files.

Writes:
  * validation_predictions.csv          (repo root, 12,000 rows: load_id,predicted_rate)
  * data/december_chart_inputs.csv      (in place, predicted_rate column filled)

Run afterwards to validate and produce the chart:

    python score.py --predictions validation_predictions.csv \\
                    --december-predictions data/december_chart_inputs.csv

The model is retrained here rather than loaded from a pickle. Training takes
under a minute and a fresh fit removes any chance of the committed predictions
disagreeing with the committed code.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import os

from src import data
from src.features import build_features, fit_feature_context, rate_from_prediction
from src.model import train_final

# score.py enforces both of these, so failing here gives a clearer message than
# failing there.
EXPECTED_VALIDATION_ROWS = 12_000
EXPECTED_DECEMBER_ROWS = 31

PREDICTIONS_FILE = data.PROJECT_ROOT / "validation_predictions.csv"


def predict_validation(model, valid: pd.DataFrame, context) -> pd.DataFrame:
    """Score every validation load and align to the provided template.

    The template is the source of truth for row order and membership: predictions
    are merged onto it rather than the other way round, so a missing or extra
    load_id surfaces as a null here instead of as a scorer rejection.
    """
    features = build_features(valid, context)
    predicted = rate_from_prediction(model.predict(features), valid["distance"])

    predictions = pd.DataFrame(
        {"load_id": valid["load_id"].to_numpy(), "predicted_rate": np.round(predicted, 2)}
    )

    template = pd.read_csv(data.TEMPLATE_FILE, usecols=["load_id"])
    merged = template.merge(predictions, on="load_id", how="left")

    if merged["predicted_rate"].isna().any():
        missing = merged.loc[merged["predicted_rate"].isna(), "load_id"].tolist()
        raise ValueError(f"no prediction produced for {len(missing)} loads, e.g. {missing[:5]}")
    if len(merged) != EXPECTED_VALIDATION_ROWS:
        raise ValueError(f"expected {EXPECTED_VALIDATION_ROWS:,} rows, got {len(merged):,}")
    if (merged["predicted_rate"] <= 0).any():
        raise ValueError("produced a non-positive rate; the scorer rejects these")

    return merged


def predict_december(model, valid: pd.DataFrame, context) -> pd.DataFrame:
    """Fill the December chart file without disturbing its other columns.

    The file is read without date parsing so the original text is written back
    unchanged; the scorer checks all seven columns, their order, and the fixed
    values, and rounding a date through pandas is an easy way to break that.

    December has no market_index of its own. It is recovered from validation.csv,
    which spans 1 Nov to 31 Dec and carries the figure for roughly 200 loads per
    December day.
    """
    december = pd.read_csv(data.DECEMBER_FILE)

    if len(december) != EXPECTED_DECEMBER_ROWS:
        raise ValueError(f"expected {EXPECTED_DECEMBER_ROWS} rows, got {len(december)}")

    parsed = december.copy()
    parsed["date"] = pd.to_datetime(parsed["date"])

    daily_market = data.daily_market_index(valid)
    missing_dates = set(parsed["date"]) - set(daily_market.index)
    if missing_dates:
        raise ValueError(f"no market index available for {sorted(missing_dates)}")

    features = build_features(parsed, context, daily_market=daily_market)
    predicted = rate_from_prediction(model.predict(features), parsed["distance"])

    december["predicted_rate"] = np.round(predicted, 2)
    return december


def _report_december(december: pd.DataFrame) -> None:
    """Print the diagnostics worth eyeballing before the chart is generated."""
    rates = december["predicted_rate"]
    spread = 100 * (rates.max() - rates.min()) / rates.mean()

    print(f"  mean ${rates.mean():.2f}   min ${rates.min():.2f}   max ${rates.max():.2f}")
    print(f"  day-to-day spread: {spread:.1f}% of the mean")
    print(f"  implied rate per mile: ${rates.mean() / december['distance'].iloc[0]:.3f}")

    # A flat line is the classic failure mode: it means the model could not
    # extrapolate and every December day collapsed to the same prediction.
    if spread < 0.1:
        print("  WARNING: predictions are essentially flat across December.")
    else:
        weekday_mean = rates[pd.to_datetime(december["date"]).dt.dayofweek < 5].mean()
        weekend_mean = rates[pd.to_datetime(december["date"]).dt.dayofweek >= 5].mean()
        print(f"  weekday mean ${weekday_mean:.2f} vs weekend mean ${weekend_mean:.2f}")


def main() -> None:
    train, valid, _ = data.load_prepared()
    context = fit_feature_context(train, data.city_coordinates(train, valid))

    print(f"training final model on {(~train.is_corrupted).sum():,} clean rows...")
    model = train_final(train, context)

    print("\nvalidation predictions")
    predictions = predict_validation(model, valid, context)
    predictions.to_csv(PREDICTIONS_FILE, index=False)
    rates = predictions["predicted_rate"]
    print(f"  {len(predictions):,} rows written to {PREDICTIONS_FILE.name}")
    print(f"  mean ${rates.mean():.2f}   median ${rates.median():.2f}   "
          f"range ${rates.min():.2f} to ${rates.max():.2f}")

    print("\nDecember chart predictions")
    december = predict_december(model, valid, context)
    december.to_csv(data.DECEMBER_FILE, index=False)
    _report_december(december)
    print(f"  written to {data.DECEMBER_FILE.relative_to(data.PROJECT_ROOT)}")

# Suppressed when invoked from run.sh / run.ps1, which runs the scorer itself.
    if os.environ.get("PIPELINE_RUN") != "1":
        print("\nNow run:")
        print("  python score.py --predictions validation_predictions.csv "
              "--december-predictions data/december_chart_inputs.csv")



if __name__ == "__main__":
    main()


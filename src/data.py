"""Loading and cleaning for the freight rate dataset.

Implements the repairs identified in notebooks/01_eda.ipynb:

  * negative `weight` values are sign errors  -> take the absolute value
  * missing `weight`                          -> median imputation
  * missing `market_index`                    -> that day's mean, then global median
  * ~1.4% of training targets are corrupted   -> flagged and dropped from TRAINING ONLY

Imputation statistics are fitted on the training set and reused for every other
frame, so validation and December predictions never depend on values a deployed
model would not have had.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- paths
# parents[1] resolves to the repository root regardless of the working directory,
# so the modules behave the same whether run via `python -m src.x` or imported
# from a notebook one level down.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

TRAIN_FILE = DATA_DIR / "train_test.csv"
VALIDATION_FILE = DATA_DIR / "validation.csv"
TEMPLATE_FILE = DATA_DIR / "validation_predictions_template.csv"
DECEMBER_FILE = DATA_DIR / "december_chart_inputs.csv"

# --------------------------------------------------------------------------- constants
# A load whose rate per mile sits beyond this factor from its lane's median is
# treated as corrupted. Chosen from the empirical gap in |log(rate ratio)|: the
# clean core dies out at 0.25 and the injected clusters do not begin until 0.75,
# so a factor of 2 (log 0.69) separates them cleanly. See EDA section 6.
CORRUPTION_THRESHOLD = 2.0

# Lanes with fewer loads than this have too thin a median to judge against, so
# their rows are never flagged. Affects ~300 of 48,000 rows.
MIN_LANE_SIZE = 3

# The holdout period for internal validation. Training data runs Jan-Oct; the real
# task is to predict Nov-Dec. Holding out Sep-Oct reproduces that forward gap.
HOLDOUT_START = "2025-09-01"


@dataclass(frozen=True)
class CleaningStats:
    """Imputation constants fitted on the training set.

    Frozen so a fitted instance cannot be mutated accidentally between the
    training and prediction paths, which would silently break reproducibility.
    """

    median_weight: float
    median_market_index: float


def fit_cleaning_stats(train: pd.DataFrame) -> CleaningStats:
    """Derive imputation constants from the training set only.

    Weight is taken in absolute value first, so the 292 sign-flipped rows
    contribute their true magnitudes to the median rather than dragging it down.
    """
    return CleaningStats(
        median_weight=float(train["weight"].abs().median()),
        median_market_index=float(train["market_index"].median()),
    )


def load_raw(path: Path) -> pd.DataFrame:
    """Read a dataset CSV with the date column parsed."""
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Place the provided CSV files in {DATA_DIR}."
        )
    return pd.read_csv(path, parse_dates=["date"])


def clean(frame: pd.DataFrame, stats: CleaningStats) -> pd.DataFrame:
    """Apply every repair identified in the EDA. Safe to call on any frame.

    Returns a copy; the input is never modified in place.
    """
    df = frame.copy()

    # --- weight: sign flips, then missing values -----------------------------
    # The negative values span exactly the same 5,000-47,500 lb range as the
    # valid ones and price identically, so the magnitude is trustworthy and only
    # the sign is wrong. Repairing beats dropping: 145 validation loads carry a
    # negative weight and still require a prediction.
    df["weight"] = df["weight"].abs()
    df["weight"] = df["weight"].fillna(stats.median_weight)

    # --- market_index: daily mean, then global median ------------------------
    # market_index is a market-wide daily figure with per-load measurement noise
    # (within-day variation is only ~15% of total). A missing value is therefore
    # best filled from that same day's other loads. The global median is a
    # fallback for any date with no observations at all.
    if "market_index" in df.columns:
        daily_mean = df.groupby("date")["market_index"].transform("mean")
        df["market_index"] = df["market_index"].fillna(daily_mean)
        df["market_index"] = df["market_index"].fillna(stats.median_market_index)

    return df


def add_lane_and_rate_per_mile(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the lane label and, where the target exists, the per-mile rate."""
    df = frame.copy()
    df["lane"] = df["pickup"] + " > " + df["delivery"]
    if "posted_rate" in df.columns:
        df["rate_per_mile"] = df["posted_rate"] / df["distance"]
    return df


def flag_corrupted_targets(
    train: pd.DataFrame, threshold: float = CORRUPTION_THRESHOLD
) -> pd.Series:
    """Return a boolean mask marking rows whose `posted_rate` looks corrupted.

    Each load's rate per mile is compared against the median for its own lane.
    Comparing within a lane controls for distance, geography and lane economics
    at once, which no global threshold can do.

    Only meaningful on labelled data. The mask must never be applied to the
    validation set: every validation row needs a prediction regardless of
    whether its features look unusual.
    """
    if "rate_per_mile" not in train.columns:
        raise ValueError("call add_lane_and_rate_per_mile() before flagging")

    lane_median = train.groupby("lane")["rate_per_mile"].transform("median")
    lane_size = train.groupby("lane")["rate_per_mile"].transform("size")
    ratio = train["rate_per_mile"] / lane_median

    outside_band = (ratio > threshold) | (ratio < 1 / threshold)
    return outside_band & (lane_size >= MIN_LANE_SIZE)


def city_coordinates(*frames: pd.DataFrame) -> pd.DataFrame:
    """Build a city -> (latitude, longitude) lookup from any number of frames.

    Every city maps to exactly one coordinate pair, and the pickup and delivery
    columns agree, so the two can be stacked. Needed for the December chart,
    whose input file supplies city names but no coordinates.
    """
    parts = []
    for frame in frames:
        parts.append(
            frame[["pickup", "pickup_lat", "pickup_lon"]].rename(
                columns={"pickup": "city", "pickup_lat": "lat", "pickup_lon": "lon"}
            )
        )
        parts.append(
            frame[["delivery", "delivery_lat", "delivery_lon"]].rename(
                columns={"delivery": "city", "delivery_lat": "lat", "delivery_lon": "lon"}
            )
        )
    stacked = pd.concat(parts, ignore_index=True).dropna()
    return stacked.drop_duplicates(subset="city").set_index("city").sort_index()


def daily_market_index(frame: pd.DataFrame) -> pd.Series:
    """Average `market_index` to one value per date.

    Used to recover the December market level, which `december_chart_inputs.csv`
    omits but `validation.csv` supplies at roughly 200 loads per day.
    """
    return frame.groupby("date")["market_index"].mean().sort_index()


def time_based_split(
    train: pd.DataFrame, holdout_start: str = HOLDOUT_START
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the labelled data chronologically.

    A random split would leak future information: the real task predicts two
    months beyond the end of training, so internal validation must do the same.
    Returns (fit, holdout).
    """
    cutoff = pd.Timestamp(holdout_start)
    fit = train[train["date"] < cutoff].copy()
    holdout = train[train["date"] >= cutoff].copy()
    return fit, holdout


def load_prepared() -> tuple[pd.DataFrame, pd.DataFrame, CleaningStats]:
    """Load, clean and annotate both labelled and unlabelled datasets.

    The corruption mask is attached as a `is_corrupted` column rather than
    applied, so callers decide where dropping is appropriate.
    """
    train = load_raw(TRAIN_FILE)
    valid = load_raw(VALIDATION_FILE)

    stats = fit_cleaning_stats(train)

    train = add_lane_and_rate_per_mile(clean(train, stats))
    valid = add_lane_and_rate_per_mile(clean(valid, stats))

    train["is_corrupted"] = flag_corrupted_targets(train)
    return train, valid, stats


if __name__ == "__main__":
    # Smoke test: python -m src.data
    train_df, valid_df, cleaning_stats = load_prepared()

    print(f"train {train_df.shape}   valid {valid_df.shape}")
    print(f"cleaning stats: {cleaning_stats}")
    print(f"negative weights remaining : {(train_df.weight < 0).sum()}")
    print(f"missing values remaining   : {train_df[['weight', 'market_index']].isna().sum().sum()}")
    print(
        f"corrupted targets flagged  : {train_df.is_corrupted.sum():,} "
        f"({100 * train_df.is_corrupted.mean():.2f}%)"
    )

    fit_df, holdout_df = time_based_split(train_df)
    print(
        f"\nsplit -> fit {len(fit_df):,} rows "
        f"({fit_df.date.min().date()} to {fit_df.date.max().date()})"
    )
    print(
        f"      holdout {len(holdout_df):,} rows "
        f"({holdout_df.date.min().date()} to {holdout_df.date.max().date()})"
    )

    coords = city_coordinates(train_df, valid_df)
    print(f"\ncity coordinate lookup: {len(coords)} cities")
    print(coords.loc[["Lexington", "Fort Wayne"]].to_string())


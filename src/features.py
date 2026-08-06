"""Feature engineering for the freight rate model.

Every design choice here follows directly from notebooks/01_eda.ipynb:

  * target is log(rate per mile), not the raw rate        (EDA section 2)
  * distance enters as a logarithm                        (EDA section 3, power law)
  * geography is carried by coordinates, not city names   (EDA section 4, unseen cities)
  * market_index is averaged to a daily figure            (EDA section 7, per-load noise)
  * quote_signal is recoded as absolute deviation         (EDA section 7, U-shape)
  * seasonality is encoded as harmonics, never a date     (EDA section 8, extrapolation)

The same function builds features for training, validation and the December
chart, so the three paths cannot silently drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Column order is fixed and shared by every caller. LightGBM keys feature
# importances by position, so a stable order keeps model artefacts comparable.
FEATURE_COLUMNS = [
    "log_distance",
    "log_weight",
    "log_market_index",
    "quote_deviation",
    "is_flatbed",
    "is_reefer",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "mid_lat",
    "mid_lon",
    "delta_lat",
    "delta_lon",
    "circuity",
    "sin_year",
    "cos_year",
    "sin_year2",
    "cos_year2",
]

EARTH_RADIUS_MILES = 3958.8


@dataclass(frozen=True)
class FeatureContext:
    """Constants fitted on training data and reused for every other frame.

    Holding these in one object makes the training/prediction contract explicit:
    if a value is not in here, it was not learned from training data.
    """

    quote_center: float
    median_quote_deviation: float
    city_coords: pd.DataFrame


def fit_feature_context(train: pd.DataFrame, city_coords: pd.DataFrame) -> FeatureContext:
    """Derive the feature-engineering constants from the training set."""
    center = float(train["quote_signal"].median())
    return FeatureContext(
        quote_center=center,
        median_quote_deviation=float((train["quote_signal"] - center).abs().median()),
        city_coords=city_coords,
    )


def haversine_miles(
    lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series
) -> pd.Series:
    """Great-circle distance in miles between two coordinate pairs."""
    lat1, lon1, lat2, lon2 = (np.radians(x) for x in (lat1, lon1, lat2, lon2))
    inner = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(inner))


def annual_harmonics(dates: pd.Series) -> pd.DataFrame:
    """First and second harmonics of the annual cycle.

    Unlike a raw date ordinal or month integer, these are defined and continuous
    for any date. A gradient-boosted tree given a date ordinal cannot split
    beyond its last training value, so every November and December load would
    inherit whatever it learned for late October. Harmonics let the fitted
    seasonal shape carry forward into the prediction window.
    """
    angle = 2 * np.pi * dates.dt.dayofyear / 365.25
    return pd.DataFrame(
        {
            "sin_year": np.sin(angle),
            "cos_year": np.cos(angle),
            "sin_year2": np.sin(2 * angle),
            "cos_year2": np.cos(2 * angle),
        },
        index=dates.index,
    )


def _resolve_coordinates(frame: pd.DataFrame, context: FeatureContext) -> pd.DataFrame:
    """Fill missing coordinates from the city lookup.

    The December chart file supplies city names but no coordinates, so they are
    resolved here rather than in a separate code path.
    """
    df = frame.copy()
    for prefix in ("pickup", "delivery"):
        lat_col, lon_col = f"{prefix}_lat", f"{prefix}_lon"
        looked_up_lat = df[prefix].map(context.city_coords["lat"])
        looked_up_lon = df[prefix].map(context.city_coords["lon"])
        df[lat_col] = df[lat_col].fillna(looked_up_lat) if lat_col in df.columns else looked_up_lat
        df[lon_col] = df[lon_col].fillna(looked_up_lon) if lon_col in df.columns else looked_up_lon

    missing = df[["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]].isna()
    if missing.any().any():
        unknown = sorted(
            set(df.loc[missing["pickup_lat"], "pickup"])
            | set(df.loc[missing["delivery_lat"], "delivery"])
        )
        raise ValueError(f"no coordinates available for: {unknown}")
    return df


def build_features(
    frame: pd.DataFrame,
    context: FeatureContext,
    daily_market: pd.Series | None = None,
) -> pd.DataFrame:
    """Build the model matrix for any frame.

    Parameters
    ----------
    frame
        Cleaned data. Must carry `pickup`, `delivery`, `distance`, `equipment`,
        `weight` and `date`. Coordinates, `market_index` and `quote_signal` are
        filled in when absent.
    context
        Constants fitted on training data.
    daily_market
        Date-indexed market index. Required when `frame` has no `market_index`
        column of its own, which is the case for the December chart inputs.

    Returns
    -------
    DataFrame with exactly FEATURE_COLUMNS, in that order.
    """
    df = _resolve_coordinates(frame, context)

    # --- market index --------------------------------------------------------
    # Always reduced to a daily figure. Within-day variation is measurement noise
    # (~15% of total variation), and averaging it away means the training,
    # validation and December paths all consume the feature in the same form.
    if daily_market is not None:
        market = df["date"].map(daily_market)
    elif "market_index" in df.columns:
        market = df.groupby("date")["market_index"].transform("mean")
    else:
        raise ValueError("frame has no market_index; pass daily_market explicitly")

    if market.isna().any():
        raise ValueError("market index unresolved for some dates")

    # --- quote signal --------------------------------------------------------
    # The raw value has no monotonic relationship with price: only its distance
    # from the centre matters, and only in the tails. Recoding as an absolute
    # deviation turns a U-shape into something a model can use directly.
    #
    # The December chart describes one hypothetical load per day, so no row-level
    # quote signal exists for it. Substituting the training median deviation asks
    # the model for the typical load on that date, which is what the chart wants.
    if "quote_signal" in df.columns:
        quote_deviation = (df["quote_signal"] - context.quote_center).abs()
        quote_deviation = quote_deviation.fillna(context.median_quote_deviation)
    else:
        quote_deviation = pd.Series(
            context.median_quote_deviation, index=df.index, dtype=float
        )

    # --- assemble ------------------------------------------------------------
    great_circle = haversine_miles(
        df["pickup_lat"], df["pickup_lon"], df["delivery_lat"], df["delivery_lon"]
    )

    features = pd.DataFrame(index=df.index)
    features["log_distance"] = np.log(df["distance"])
    features["log_weight"] = np.log(df["weight"])
    features["log_market_index"] = np.log(market)
    features["quote_deviation"] = quote_deviation

    # Dry Van is the reference level, so two indicators cover three categories.
    features["is_flatbed"] = (df["equipment"] == "Flatbed").astype(float)
    features["is_reefer"] = (df["equipment"] == "Reefer").astype(float)

    # Raw coordinates let the model place an unseen city among its neighbours.
    features["pickup_lat"] = df["pickup_lat"]
    features["pickup_lon"] = df["pickup_lon"]
    features["delivery_lat"] = df["delivery_lat"]
    features["delivery_lon"] = df["delivery_lon"]

    # Midpoint locates the lane as a whole; the deltas encode direction and
    # orientation, which raw endpoints only express through interactions.
    features["mid_lat"] = (df["pickup_lat"] + df["delivery_lat"]) / 2
    features["mid_lon"] = (df["pickup_lon"] + df["delivery_lon"]) / 2
    features["delta_lat"] = df["delivery_lat"] - df["pickup_lat"]
    features["delta_lon"] = df["delivery_lon"] - df["pickup_lon"]

    # Road distance divided by great-circle distance: how indirect the routing
    # is. Typically ~1.18; higher values indicate detours around terrain.
    features["circuity"] = df["distance"] / great_circle.replace(0, np.nan)
    features["circuity"] = features["circuity"].fillna(1.0)

    features = pd.concat([features, annual_harmonics(df["date"])], axis=1)

    return features[FEATURE_COLUMNS]


def build_target(frame: pd.DataFrame) -> pd.Series:
    """Model target: log of rate per mile.

    Predictions are converted back with `exp(prediction) * distance`. Working in
    this space bakes in the near-proportionality to distance, makes errors
    multiplicative to match how freight prices behave, and stabilises variance
    across a target spanning two orders of magnitude.
    """
    return np.log(frame["posted_rate"] / frame["distance"])


def rate_from_prediction(
    log_rate_per_mile: np.ndarray, distance: pd.Series
) -> np.ndarray:
    """Convert a model prediction back to dollars."""
    return np.exp(log_rate_per_mile) * distance.to_numpy()


if __name__ == "__main__":
    # Smoke test: python -m src.features
    from src import data

    train, valid, _ = data.load_prepared()
    coords = data.city_coordinates(train, valid)
    context = fit_feature_context(train, coords)

    X_train = build_features(train, context)
    X_valid = build_features(valid, context)
    print(f"train matrix {X_train.shape}   valid matrix {X_valid.shape}")
    print(f"NaNs -> train {int(X_train.isna().sum().sum())}, valid {int(X_valid.isna().sum().sum())}")

    # The December path exercises every fallback at once: no coordinates,
    # no market_index, no quote_signal.
    december = data.load_raw(data.DECEMBER_FILE)
    X_december = build_features(
        december, context, daily_market=data.daily_market_index(valid)
    )
    print(f"december matrix {X_december.shape}   NaNs {int(X_december.isna().sum().sum())}")
    print(f"market index varies across December: {X_december.log_market_index.std():.4f}")


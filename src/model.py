"""Model selection, validation and training.

Validation is rolling-origin with two-month test blocks, mirroring the real task
(train through October, predict November and December). A random split would leak
future information and report a score the model cannot reproduce on the real
validation set.

Metrics are reported on clean rows and on all rows separately. Roughly 1.4% of
labels are corrupted by a multiplicative factor; no model can predict those, so
the clean-row figure measures modelling quality while the all-row figure shows
what a metric computed over the raw labels would look like.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src import data
from src.features import (
    FeatureContext,
    build_features,
    build_target,
    rate_from_prediction,
)

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    LIGHTGBM_AVAILABLE = False

# Chosen by sweeping against the rolling-origin folds below, not against a single
# holdout. Subsampling and column sampling cost ~1 MAE point on the easiest fold
# but cut the spread across folds roughly in half, which matters more when the
# real test period sits beyond anything we can measure.
MODEL_PARAMS = {
    "n_estimators": 1500,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_child_samples": 40,
    "colsample_bytree": 0.7,
    "subsample": 0.8,
    "subsample_freq": 1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

# Each tuple is (test_start, test_end). Training uses everything before the start,
# so the window expands with each fold while the test block stays two months.
CV_FOLDS = [
    ("2025-05-01", "2025-07-01"),
    ("2025-07-01", "2025-09-01"),
    ("2025-09-01", "2025-11-01"),
]


def evaluate(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Error metrics in dollars and percent.

    The assessment does not state which metric Spotter uses, so several are
    reported. MAE and MAPE reward the conditional median, RMSE the conditional
    mean; with residuals this tight the two barely diverge, but reporting both
    makes the choice explicit rather than accidental.
    """
    absolute_error = np.abs(predicted - actual)
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAPE_%": float(100 * np.mean(absolute_error / actual)),
        "MedAPE_%": float(100 * np.median(absolute_error / actual)),
    }


def make_model(name: str):
    """Return an untrained estimator for the log(rate per mile) target."""
    if name == "ridge":
        return Ridge(alpha=1.0)
    if name == "lightgbm":
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("lightgbm is not installed; run pip install -r requirements.txt")
        return lgb.LGBMRegressor(**MODEL_PARAMS)
    raise ValueError(f"unknown model: {name}")


def _fold_frames(train: pd.DataFrame, start: str, end: str):
    """Split one fold. Corrupted targets are dropped from the fit side only."""
    fit = train[(train["date"] < pd.Timestamp(start)) & (~train["is_corrupted"])]
    test = train[(train["date"] >= pd.Timestamp(start)) & (train["date"] < pd.Timestamp(end))]
    return fit, test


def cross_validate(
    train: pd.DataFrame, context: FeatureContext, model_name: str
) -> pd.DataFrame:
    """Rolling-origin evaluation. Returns one row of metrics per fold."""
    rows = []
    for start, end in CV_FOLDS:
        fit, test = _fold_frames(train, start, end)

        model = make_model(model_name)
        model.fit(build_features(fit, context), build_target(fit))

        predicted = rate_from_prediction(
            model.predict(build_features(test, context)), test["distance"]
        )
        actual = test["posted_rate"].to_numpy()
        clean = (~test["is_corrupted"]).to_numpy()

        metrics = evaluate(actual[clean], predicted[clean])
        metrics["MAE_all_rows"] = float(mean_absolute_error(actual, predicted))
        metrics["fold"] = f"{start[:7]} to {end[:7]}"
        metrics["n_test"] = len(test)
        rows.append(metrics)

    return pd.DataFrame(rows).set_index("fold")


def naive_baseline_metrics(train: pd.DataFrame) -> pd.DataFrame:
    """Constant rate-per-mile baseline: every load priced at the median $/mile.

    Included so the modelling gain is quoted against something, rather than an
    unstated assumption about what "good" means.
    """
    rows = []
    for start, end in CV_FOLDS:
        fit, test = _fold_frames(train, start, end)
        predicted = fit["rate_per_mile"].median() * test["distance"].to_numpy()

        actual = test["posted_rate"].to_numpy()
        clean = (~test["is_corrupted"]).to_numpy()

        metrics = evaluate(actual[clean], predicted[clean])
        metrics["MAE_all_rows"] = float(mean_absolute_error(actual, predicted))
        metrics["fold"] = f"{start[:7]} to {end[:7]}"
        metrics["n_test"] = len(test)
        rows.append(metrics)

    return pd.DataFrame(rows).set_index("fold")


def train_final(train: pd.DataFrame, context: FeatureContext, model_name: str = "lightgbm"):
    """Fit on every clean labelled row.

    Validation runs to 31 December, so the final model uses the full January to
    October window rather than holding anything back. Hyperparameters were fixed
    beforehand by cross-validation, so nothing is being tuned on data the model
    now trains on.
    """
    clean = train[~train["is_corrupted"]]
    model = make_model(model_name)
    model.fit(build_features(clean, context), build_target(clean))
    return model


def feature_importances(model, context: FeatureContext) -> pd.Series:
    """Gain-based feature importance, normalised to percentages."""
    from src.features import FEATURE_COLUMNS

    if hasattr(model, "booster_"):
        gains = model.booster_.feature_importance(importance_type="gain")
    else:
        gains = np.abs(getattr(model, "coef_", np.zeros(len(FEATURE_COLUMNS))))

    series = pd.Series(gains, index=FEATURE_COLUMNS)
    return (100 * series / series.sum()).sort_values(ascending=False)


def _summarise(name: str, folds: pd.DataFrame) -> dict[str, object]:
    """Collapse per-fold metrics into a single row for the comparison table."""
    return {
        "model": name,
        "MAE": folds["MAE"].mean(),
        "RMSE": folds["RMSE"].mean(),
        "MAPE_%": folds["MAPE_%"].mean(),
        "MedAPE_%": folds["MedAPE_%"].mean(),
        "MAE_worst_fold": folds["MAE"].max(),
        "MAE_all_rows": folds["MAE_all_rows"].mean(),
    }


def main() -> None:
    """Compare baselines, report validation metrics, save artefacts."""
    from src.features import fit_feature_context

    train, valid, _ = data.load_prepared()
    context = fit_feature_context(train, data.city_coordinates(train, valid))

    print("Rolling-origin validation, two-month test blocks")
    print(f"training rows: {len(train):,}   corrupted excluded from fits: {train.is_corrupted.sum():,}\n")

    summaries = [_summarise("median $/mile", naive_baseline_metrics(train))]

    ridge_folds = cross_validate(train, context, "ridge")
    print("Ridge, per fold:")
    print(ridge_folds.round(2).to_string(), "\n")
    summaries.append(_summarise("ridge", ridge_folds))

    if LIGHTGBM_AVAILABLE:
        lgbm_folds = cross_validate(train, context, "lightgbm")
        print("LightGBM, per fold:")
        print(lgbm_folds.round(2).to_string(), "\n")
        summaries.append(_summarise("lightgbm", lgbm_folds))
    else:
        print("LightGBM unavailable; skipping. Ridge will be used as the final model.\n")

    comparison = pd.DataFrame(summaries).set_index("model").round(2)
    print("Mean across folds:")
    print(comparison.to_string())

    data.OUTPUT_DIR.mkdir(exist_ok=True)
    comparison.to_csv(data.OUTPUT_DIR / "model_comparison.csv")

    chosen = "lightgbm" if LIGHTGBM_AVAILABLE else "ridge"
    model = train_final(train, context, chosen)

    importances = feature_importances(model, context)
    importances.to_csv(data.OUTPUT_DIR / "feature_importance.csv", header=["gain_pct"])
    print(f"\nTop features for {chosen} (% of total gain):")
    print(importances.head(10).round(2).to_string())

    _plot_importances(importances, data.OUTPUT_DIR / "fig_feature_importance.png")
    print(f"\nwrote {data.OUTPUT_DIR / 'model_comparison.csv'}")
    print(f"wrote {data.OUTPUT_DIR / 'fig_feature_importance.png'}")


def _plot_importances(importances: pd.Series, path) -> None:
    """Horizontal bar chart of the top features, for the report."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = importances.head(12).iloc[::-1]
    figure, axis = plt.subplots(figsize=(8, 5), dpi=150)
    axis.barh(top.index, top.to_numpy(), color="#064A56")
    axis.set_xlabel("share of total gain (%)")
    axis.set_title("Feature importance", loc="left", fontweight="bold")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()


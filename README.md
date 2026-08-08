# Freight Rate Prediction

Predicts posted freight rates for US truckload shipments from lane, equipment,
weight, date and market-condition features.

Submission for the Spotter Machine Learning Engineer assessment.

## Results

Rolling-origin validation, three folds with two-month test blocks. Metrics are
means across folds, computed on rows with uncorrupted labels.

| Model | MAE | RMSE | MAPE | Worst fold (MAE) |
|---|---|---|---|---|
| Median $/mile baseline | $198.08 | $284.11 | 9.40% | $203.69 |
| Ridge regression | $147.53 | $206.24 | 6.12% | $238.75 |
| **LightGBM** | **$61.09** | **$112.40** | **2.56%** | **$64.22** |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

requirements-lock.txt records the exact versions these results were produced with; requirements.txt gives the supported ranges. 

The provided data files are not committed to this repository. Place these three
in `data/` before running:

- `train_test.csv`
- `validation.csv`
- `validation_predictions_template.csv`

`data/december_chart_inputs.csv` is committed, since its `predicted_rate` column
is part of the submission.

## Usage

```bash
python -m src.model      # validation metrics and model comparison  (~90s)
python -m src.predict    # writes both submission files             (~40s)
python score.py --predictions validation_predictions.csv \
                --december-predictions data/december_chart_inputs.csv
```

Or run all three at once: `bash run.sh` / `.\run.ps1`.

Each module also has a standalone smoke test: `python -m src.data`,
`python -m src.features`.

## Approach

### Validation strategy

Training data covers 1 Jan – 31 Oct 2025; the validation set covers 1 Nov –
31 Dec with no overlap. This is a forecasting problem, so a random split would
leak future information and overstate performance.

Internal validation is therefore rolling-origin with expanding training windows
and fixed two-month test blocks, reproducing the real forward gap:

| Fold | Trains on | Tests on |
|---|---|---|
| 1 | Jan – Apr | May – Jun |
| 2 | Jan – Jun | Jul – Aug |
| 3 | Jan – Aug | Sep – Oct |

Hyperparameters were selected against the mean across all three folds rather
than a single holdout. The final model then refits on the complete Jan – Oct
window.

### Data quality

Five issues were identified during exploration (`notebooks/01_eda.ipynb`) and
are handled in `src/data.py`:

| Issue | Train | Validation | Treatment |
|---|---|---|---|
| Negative `weight` | 292 | 145 | Absolute value — sign flip, magnitudes valid |
| Missing `weight` | 300 | 165 | Median imputation |
| Missing `market_index` | 374 | 249 | That day's mean, then global median |
| Corrupted `posted_rate` | 670 | n/a | Excluded from training only |
| Cities absent from training | — | 8 | Geography encoded by coordinate |

The corrupted rates are the most consequential. Comparing each load's rate per
mile against its own lane's median produces a tight unimodal core flanked by two
detached clusters at roughly ×2–6 and ÷2–6, with an empty gap between them —
the signature of injected noise rather than a heavy natural tail. The threshold
was placed inside that observed gap rather than at an arbitrary percentile.

### Features

Nineteen features, built by a single function shared by the training,
validation and December paths so they cannot drift apart.

The target is `log(rate per mile)`, reconstructed as `exp(prediction) × distance`.
Rates follow a clean power law (`rate ≈ distance^0.874`), so this bakes in the
distance relationship, makes errors multiplicative, and stabilises variance over
a target spanning two orders of magnitude.

Three choices worth calling out:

- **Coordinates, not city names.** Eight cities appear only in validation,
  touching 12% of its rows. Holding out eight cities from training entirely,
  coordinate features scored $30.90 MAE against $34.78 for integer city IDs.
- **Day-of-year harmonics for seasonality.** Defined and continuous beyond the
  training window, unlike a date ordinal. Removing seasonality entirely costs
  4.2 MAE overall, but the damage concentrates in the furthest-out fold
  ($62.29 → $85.42) — precisely the regime December occupies.
- **`quote_signal` recoded as absolute deviation from its median.** The raw
  value has no monotonic relationship with price; only its distance from the
  centre matters.

### Model

LightGBM regressing on `log(rate per mile)`. Subsampling and column sampling
cost about 1 MAE point on the easiest fold but roughly halve the spread across
folds, which matters more when the real test period lies beyond anything
measurable here.

Feature ablations, mean MAE across folds:

| Removed | MAE | Cost |
|---|---|---|
| Nothing | $61.09 | — |
| `quote_deviation` | $71.96 | +10.9 |
| `log_market_index` | $67.52 | +6.4 |
| Seasonal harmonics | $65.30 | +4.2 |
| *Corruption filter disabled* | $84.32 | +23.2 |

Dropping corrupted rows is worth more than every feature decision combined.

### December chart

`december_chart_inputs.csv` supplies only seven columns — no `market_index`, no
`quote_signal`, no coordinates. `validation.csv` spans 1 Nov – 31 Dec and does
carry `market_index` for all 31 December days at roughly 200 loads per day, so
the daily market level is recovered by averaging it there. Coordinates come from
the city lookup built across both labelled files.

The result shows the weekly market cycle carried into December, with weekdays
averaging $810.12 against $805.65 at weekends. The predicted level, $2.247 per
mile, sits just below October's $2.207 and well above January's $2.068 — a mild
seasonal decline rather than a full reversion to the winter trough.

## Project structure

| Path | Purpose |
|---|---|
| `src/data.py` | Loading, cleaning, corruption detection, time-based split |
| `src/features.py` | Feature engineering for all three prediction paths |
| `src/model.py` | Rolling-origin validation, baselines, final training |
| `src/predict.py` | Writes both submission files |
| `notebooks/01_eda.ipynb` | Exploratory analysis, ten sections |
| `outputs/` | Figures, model comparison, feature importances |
| `validation_predictions.csv` | **Deliverable** — 12,000 predictions |
| `data/december_chart_inputs.csv` | **Deliverable** — 31 December predictions |
| `scorer_results/candidate_december.png` | Chart produced by `score.py` |
| `score.py` | Provided scorer, unmodified |

## Reproducibility

All random seeds fixed at 42. `src/predict.py` retrains from scratch rather than
loading a serialised model, so the committed predictions cannot drift from the
committed code.

## Limitations

- The evaluation metric was not specified, so MAE, RMSE, MAPE and median APE are
  all reported rather than optimising for one.
- Corruption detection uses a lane median, which needs at least three loads on a
  lane to be meaningful. Roughly 300 of 48,000 rows sit on thinner lanes and
  cannot be checked this way.
- Residual scatter is about 5% and appears largely irreducible; the gap between
  a well-specified linear model and gradient boosting is real but the remaining
  headroom is small.


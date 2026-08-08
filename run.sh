#!/usr/bin/env bash
# Full pipeline: validation metrics, predictions, scorer.
# Usage: bash run.sh
set -euo pipefail

python -m src.model
python -m src.predict
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv

# Full pipeline: validation metrics, predictions, scorer.
# Usage: .\run.ps1
$ErrorActionPreference = "Stop"
$env:PIPELINE_RUN = "1"

python -m src.model
python -m src.predict
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv


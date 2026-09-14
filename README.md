# Smart Energy Optimizer

A Data Science portfolio project based on the **Smart Meters in London** dataset.

The current project uses household electricity data to:

1. forecast the next 24 hours of consumption;
2. find unusual deviations from the forecast.

Data preparation, forecasting and unusual-event detection are complete. Energy cost optimization is the next step.

## Project workflow

```text
Raw smart-meter data
        ↓
Cleaning and temporal exploration
        ↓
24-hour consumption forecast
        ↓
Unusual-event detection
        ↓
Future energy cost optimization
```

## 1. Data exploration

The project uses 24 households selected from three ACORN groups and two tariff groups. The sample is balanced for development, but it is not representative of all London households.

| Period | Use |
|---|---|
| September 2012–August 2013 | Training |
| September–October 2013 | Validation |
| November–December 2013 | Test |

The final grid contains 30-minute observations. Missing readings stay missing. Zero and high readings are kept.

Main findings:

- training coverage is 99.97%;
- average demand is highest around 18:30;
- weekend consumption is slightly higher;
- consumption and forecast error differ strongly across households;
- daily and weekly historical values are useful forecasting references.

Notebook: [01_data_exploration.ipynb](notebooks/01_data_exploration.ipynb)

## 2. Consumption forecasting

At midnight, the model predicts the next 48 half-hour values for each household.

The selected model is LightGBM with:

- consumption at the same time yesterday and last week;
- previous-day and previous-week energy totals;
- half-hour slot;
- day of week;
- weekend indicator;
- month.

Observed target-day weather was tested only as an oracle experiment. It added very little and is not available at midnight, so it was excluded from the final model.

| Test result | MAE | RMSE |
|---|---:|---:|
| Weekly naive | 0.153456 | 0.299838 |
| LightGBM | **0.124414** | **0.256759** |

LightGBM reduces test MAE by **18.9%** compared with the weekly naive baseline.

Notebook: [02_forecasting.ipynb](notebooks/02_forecasting.ipynb)

## 3. Unusual consumption events

The detector uses the difference between actual and predicted consumption:

```text
residual = actual consumption - predicted consumption
```

Residuals are normalized separately for each household using the validation median and MAD. The threshold is fitted on validation and then applied unchanged to test.

The selected threshold keeps the most unusual 0.5% of validation observations.

| Test result | Value |
|---|---:|
| Flagged half-hours | 1,014 |
| Alert rate | 1.51% |
| Higher than expected | 979 |
| Lower than expected | 35 |
| Grouped episodes | 566 |

These are **unusual-event candidates**, not confirmed anomalies. Isolation Forest was used only as a comparison. The robust residual score remains the main method because it is easier to explain.

Notebook: [03_anomaly_detection.ipynb](notebooks/03_anomaly_detection.ipynb)

## Main project files

```text
notebooks/
    01_data_exploration.ipynb
    02_forecasting.ipynb
    03_anomaly_detection.ipynb

src/
    data/               # Cleaning and feature preparation
    forecasting/        # Model training, prediction and evaluation
    anomaly/            # Robust residual scoring and episode grouping

scripts/
    run_exploration.py
    run_exploration_forecasting.py
    prepare_forecasting_data.py
    run_forecasting.py
    run_anomaly_detection.py

data/processed/         # Generated local datasets
models/                 # Generated forecasting model
outputs/                # Generated metrics and comparisons
reports/                # Short reports and figures
tests/                  # Focused checks
```

The raw dataset is stored locally in `archive/` and is excluded from Git. Generated datasets, models and technical outputs are also excluded.

## Reproduce the results

Run the commands from the project root in WSL:

```bash
python3 -m pip install -r requirements.txt

# Exploration and baselines
python3 scripts/run_exploration.py

# Forecasting
python3 scripts/prepare_forecasting_data.py
python3 scripts/run_forecasting.py

# Or rebuild exploration and forecasting together
python3 scripts/run_exploration_forecasting.py

# Unusual-event detection
python3 scripts/run_anomaly_detection.py

# Tests
python3 -m unittest discover -s tests -v
```

## Important limitations

- The 24-household sample is small and balanced by design.
- Forecast performance changes with season and household behavior.
- Unusual-event candidates have no confirmed ground-truth labels.
- Large residuals may come from behavior, model error or data issues.

## Next step

Build a simple energy cost simulation that:

- uses the forecast profile available at midnight;
- separates fixed and assumed flexible consumption;
- preserves total daily energy;
- compares baseline and optimized cost;
- clearly labels tariff and flexibility assumptions.

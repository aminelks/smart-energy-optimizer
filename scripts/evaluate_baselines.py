"""Create forecast-origin-safe features and score validation only."""
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.features import make_day_ahead_features
from src.evaluation import evaluate_baselines


def main():
    panel = pd.read_parquet(ROOT / 'data/processed/energy_halfhourly.parquet')
    features = make_day_ahead_features(panel)
    features.to_parquet(ROOT / 'data/processed/forecast_features.parquet', index=False)
    scores, households, availability = evaluate_baselines(features)
    for name, frame in [('baseline_metrics', scores), ('baseline_metrics_by_household', households),
                        ('baseline_availability', availability)]:
        frame.to_csv(ROOT / f'reports/{name}.csv', index=False)
    print(scores.to_string(index=False))
    print('Household-macro MAE:', households.groupby('baseline').mae_kwh.mean().to_dict())


if __name__ == '__main__':
    main()

"""Build residual anomaly outputs from the saved forecasting predictions."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.anomaly.detector import fit_residual_calibration, score_residuals, make_episodes


def main():
    source = pd.read_parquet(ROOT / 'data/processed/forecasting_predictions.parquet')
    source = source.rename(columns={
        'energy_kwh': 'actual_consumption',
        'predicted_kwh': 'predicted_consumption',
    })
    source['slot'] = source['timestamp'].dt.hour * 2 + source['timestamp'].dt.minute // 30
    source['day_of_week'] = source['timestamp'].dt.dayofweek
    source['weekend'] = source['day_of_week'].ge(5)

    validation = source.loc[source['split'].eq('validation')].copy()
    calibration = fit_residual_calibration(validation)
    scored = score_residuals(source, calibration)

    threshold_rows = []
    for expected_rate in [.005, .01, .02]:
        threshold = validation.merge(calibration, on='household_id', validate='many_to_one')
        threshold = (
            (threshold['residual'] - threshold['residual_median']).abs()
            / threshold['residual_scale']
        ).quantile(1 - expected_rate)
        flags = scored['anomaly_score'].ge(threshold)
        for split in ['validation', 'test']:
            mask = scored['split'].eq(split)
            counts = scored.loc[mask & flags].groupby('household_id').size()
            threshold_rows.append({
                'expected_validation_rate': expected_rate,
                'threshold': threshold,
                'split': split,
                'flagged_points': int((mask & flags).sum()),
                'alert_rate': float(flags.loc[mask].mean()),
                'households_flagged': int(len(counts)),
                'largest_household_share': float(counts.max() / counts.sum()),
            })
    thresholds = pd.DataFrame(threshold_rows)
    selected_threshold = thresholds.loc[
        thresholds['expected_validation_rate'].eq(.005), 'threshold'
    ].iloc[0]
    scored['is_anomaly'] = scored['anomaly_score'].ge(selected_threshold)

    global_threshold = validation['absolute_error'].quantile(.99)
    scored['global_is_anomaly'] = scored['absolute_error'].ge(global_threshold)

    isolation_features = ['residual', 'absolute_error', 'actual_consumption', 'slot', 'day_of_week']
    isolation_model = make_pipeline(
        StandardScaler(),
        IsolationForest(n_estimators=200, contamination=.01, random_state=42, n_jobs=-1),
    )
    isolation_model.fit(validation[isolation_features])
    scored['isolation_score'] = -isolation_model.decision_function(scored[isolation_features])
    scored['isolation_is_anomaly'] = isolation_model.predict(scored[isolation_features]) == -1

    episodes = make_episodes(scored)
    output_columns = [
        'household_id', 'timestamp', 'actual_consumption', 'predicted_consumption',
        'residual', 'absolute_error', 'anomaly_score', 'is_anomaly',
        'anomaly_direction', 'split', 'slot', 'day_of_week', 'weekend',
        'global_is_anomaly', 'isolation_score', 'isolation_is_anomaly',
    ]
    scored[output_columns].to_parquet(
        ROOT / 'data/processed/anomaly_results.parquet', index=False
    )
    episodes.to_parquet(ROOT / 'data/processed/anomaly_episodes.parquet', index=False)
    calibration.to_csv(ROOT / 'outputs/anomaly_calibration.csv', index=False)
    thresholds.to_csv(ROOT / 'outputs/anomaly_threshold_comparison.csv', index=False)

    test = scored.loc[scored['split'].eq('test')]
    residual_flags = test['is_anomaly']
    isolation_flags = test['isolation_is_anomaly']
    overlap = residual_flags & isolation_flags
    test_episodes = episodes.loc[episodes['split'].eq('test')]
    metadata = {
        'method': 'household robust residual score',
        'score': 'abs(residual - validation household median) / (1.4826 * validation household MAD)',
        'calibration_period': '2013-09-01 to 2013-10-31',
        'application_period': '2013-11-01 to 2013-12-31',
        'selected_expected_validation_alert_rate': .005,
        'selected_threshold': float(selected_threshold),
        'validation_alert_rate': float(scored.loc[scored['split'].eq('validation'), 'is_anomaly'].mean()),
        'test_alert_rate': float(residual_flags.mean()),
        'test_flagged_points': int(residual_flags.sum()),
        'test_positive_points': int((residual_flags & test['residual'].ge(0)).sum()),
        'test_negative_points': int((residual_flags & test['residual'].lt(0)).sum()),
        'test_episodes': int(len(test_episodes)),
        'test_multi_point_episodes': int(test_episodes['duration_slots'].gt(1).sum()),
        'isolation_forest_test_alert_rate': float(isolation_flags.mean()),
        'test_overlap_points': int(overlap.sum()),
        'overlap_share_of_residual_flags': float(overlap.sum() / residual_flags.sum()),
        'household_fallbacks': int(calibration['used_global_fallback'].sum()),
        'claim': 'Candidates are unusual deviations, not confirmed anomalies.',
    }
    (ROOT / 'outputs/anomaly_metrics.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    print('\nThreshold comparison')
    print(thresholds.to_string(index=False))
    print('\nTop test episodes')
    print(test_episodes.head(10).to_string(index=False))


if __name__ == '__main__':
    main()

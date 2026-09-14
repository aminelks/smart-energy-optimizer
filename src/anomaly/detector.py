"""Leakage-safe residual anomaly detection."""
import numpy as np
import pandas as pd


def fit_residual_calibration(frame, minimum_rows=100):
    """Fit robust household statistics on past residuals."""
    global_median = frame['residual'].median()
    global_mad = (frame['residual'] - global_median).abs().median()
    global_scale = max(1.4826 * global_mad, 1e-6)

    rows = []
    for household, data in frame.groupby('household_id'):
        median = data['residual'].median()
        mad = (data['residual'] - median).abs().median()
        scale = 1.4826 * mad
        use_household = len(data) >= minimum_rows and scale > 1e-6
        rows.append({
            'household_id': household,
            'calibration_rows': len(data),
            'residual_median': median if use_household else global_median,
            'residual_scale': scale if use_household else global_scale,
            'used_global_fallback': not use_household,
        })
    return pd.DataFrame(rows)


def score_residuals(frame, calibration):
    """Calculate household-normalized robust residual scores."""
    result = frame.merge(calibration, on='household_id', how='left', validate='many_to_one')
    result['anomaly_score'] = (
        (result['residual'] - result['residual_median']).abs()
        / result['residual_scale']
    )
    result['anomaly_direction'] = np.where(
        result['residual'].ge(0), 'higher_than_expected', 'lower_than_expected'
    )
    return result


def make_episodes(scored):
    """Group consecutive flagged half-hours for each household and split."""
    flagged = scored.loc[scored['is_anomaly']].sort_values(
        ['split', 'household_id', 'timestamp']
    ).copy()
    if flagged.empty:
        return pd.DataFrame()

    new_episode = (
        flagged['household_id'].ne(flagged['household_id'].shift())
        | flagged['split'].ne(flagged['split'].shift())
        | flagged['timestamp'].diff().ne(pd.Timedelta(minutes=30))
    )
    flagged['episode_id'] = new_episode.cumsum()

    rows = []
    for _, data in flagged.groupby('episode_id'):
        most_unusual = data.loc[data['anomaly_score'].idxmax()]
        rows.append({
            'household_id': data['household_id'].iloc[0],
            'split': data['split'].iloc[0],
            'episode_start': data['timestamp'].min(),
            'episode_end': data['timestamp'].max(),
            'duration_slots': len(data),
            'max_anomaly_score': data['anomaly_score'].max(),
            'max_absolute_error': data['absolute_error'].max(),
            'direction': most_unusual['anomaly_direction'],
        })
    return pd.DataFrame(rows).sort_values('max_anomaly_score', ascending=False)

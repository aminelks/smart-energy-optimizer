"""Reusable version of the feature steps introduced in notebook 01."""
import pandas as pd
from .preprocessing import VALID_START, TEST_START

HISTORY_FEATURES = ['lag_48', 'lag_336']
CALENDAR_FEATURES = ['slot', 'day_of_week', 'month']


def make_forecasting_features(panel, train_start, valid_start, test_start, end):
    """Create features available before each target day starts."""
    keys = ['household_id', 'timestamp']
    result = panel[keys + ['energy_kwh']].copy()
    result['forecast_origin'] = result['timestamp'].dt.floor('D')
    result['slot'] = result['timestamp'].dt.hour * 2 + result['timestamp'].dt.minute // 30
    result['day_of_week'] = result['timestamp'].dt.dayofweek
    result['weekend'] = result['day_of_week'].ge(5).astype(int)
    result['month'] = result['timestamp'].dt.month

    for periods in [48, 336]:
        history = panel[keys + ['energy_kwh']].copy()
        history['timestamp'] = history['timestamp'] + pd.Timedelta(minutes=30 * periods)
        history = history.rename(columns={'energy_kwh': f'lag_{periods}'})
        result = result.merge(history, on=keys, how='left', validate='one_to_one')

    daily = panel.assign(day=panel['timestamp'].dt.floor('D')).groupby(
        ['household_id', 'day']
    )['energy_kwh'].agg(['count', 'sum']).reset_index()
    daily['daily_total'] = daily['sum'].where(daily['count'].eq(48))

    for days, name in [(1, 'previous_day_total'), (7, 'previous_week_total')]:
        history = daily[['household_id', 'day', 'daily_total']].copy()
        history['forecast_origin'] = history['day'] + pd.Timedelta(days=days)
        history = history.rename(columns={'daily_total': name}).drop(columns='day')
        result = result.merge(history, on=['household_id', 'forecast_origin'], how='left', validate='many_to_one')

    result = result.loc[result['forecast_origin'].ge(train_start) & result['forecast_origin'].lt(end)].copy()
    result['split'] = 'train'
    result.loc[result['forecast_origin'].ge(valid_start), 'split'] = 'validation'
    result.loc[result['forecast_origin'].ge(test_start), 'split'] = 'test'
    return result


def add_hourly_weather(features, weather, weather_columns):
    """Add observed hourly weather for an explicitly labeled oracle experiment."""
    weather = weather[['time'] + weather_columns].copy().sort_values('time')
    hourly_index = pd.date_range(weather['time'].min(), weather['time'].max(), freq='h')
    weather = weather.set_index('time').reindex(hourly_index)
    weather[weather_columns] = weather[weather_columns].interpolate(method='time', limit_direction='both')
    weather.index.name = 'weather_hour'
    weather = weather.add_prefix('weather_').reset_index()

    result = features.copy()
    result['weather_hour'] = result['timestamp'].dt.floor('h')
    return result.merge(weather, on='weather_hour', how='left', validate='many_to_one')


def make_day_ahead_features(panel):
    """Look up exact historical timestamps, preserving missing readings as NaN.

    For target slots [D, D+1 day), both historical readings precede origin D.
    A lag_1 or rolling window ending at each target slot would leak later in the day.
    """
    keys = ['household_id', 'timestamp']
    if panel.duplicated(keys).any():
        raise ValueError('Duplicate household/timestamp pairs must be resolved first.')
    if panel.timestamp.isna().any() or panel.timestamp.ne(panel.timestamp.dt.floor('30min')).any():
        raise ValueError('Timestamps must be valid and aligned to 30 minutes.')
    result = panel[keys + ['energy_kwh']].sort_values(keys).reset_index(drop=True)
    result['forecast_origin'] = result['timestamp'].dt.floor('D')
    result['slot'] = result['timestamp'].dt.hour * 2 + result['timestamp'].dt.minute // 30
    result['day_of_week'] = result['timestamp'].dt.dayofweek
    result['month'] = result['timestamp'].dt.month
    for periods in [48, 336]:
        history = result[keys + ['energy_kwh']].copy()
        # Move yesterday's reading to today's matching slot (and likewise for last week).
        history['timestamp'] = history['timestamp'] + pd.Timedelta(minutes=30 * periods)
        history = history.rename(columns={'energy_kwh': f'lag_{periods}'})
        result = result.merge(history, on=keys, how='left', validate='one_to_one')
        source_time = result['timestamp'] - pd.Timedelta(minutes=30 * periods)
        assert (source_time < result['forecast_origin']).all()
    result['split'] = 'train'
    result.loc[result['forecast_origin'] >= VALID_START, 'split'] = 'validation'
    result.loc[result['forecast_origin'] >= TEST_START, 'split'] = 'test'
    result['evaluation_eligible'] = result[['energy_kwh'] + HISTORY_FEATURES].notna().all(axis=1)
    return result

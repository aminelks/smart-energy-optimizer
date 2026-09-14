"""Training-only temporal summaries used to choose the forecasting target."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    panel = pd.read_parquet(ROOT / 'data/processed/energy_halfhourly.parquet')
    train = panel.loc[panel.timestamp.lt('2013-09-01')].copy()
    train['slot'] = train.timestamp.dt.hour * 2 + train.timestamp.dt.minute // 30
    train['day_of_week'] = train.timestamp.dt.dayofweek
    train['weekend'] = train.day_of_week.ge(5)
    reports = ROOT / 'reports'
    profile = train.groupby(['slot', 'weekend']).energy_kwh.mean().unstack()
    profile.to_csv(reports / 'intraday_profile.csv')
    weekly = train.groupby('day_of_week').energy_kwh.mean()
    weekly.to_csv(reports / 'weekday_profile.csv')
    daily = train.groupby(['household_id', train.timestamp.dt.floor('D')]).energy_kwh.agg(['count', 'sum'])
    daily['complete_day_kwh'] = daily['sum'].where(daily['count'].eq(48))
    daily.to_csv(reports / 'training_daily_consumption.csv')
    monthly = daily.reset_index().groupby(pd.Grouper(key='timestamp', freq='MS')).complete_day_kwh.mean()
    monthly.to_csv(reports / 'monthly_profile.csv')
    wide = train.pivot(index='timestamp', columns='household_id', values='energy_kwh')
    lag_table = pd.DataFrame({lag: wide.corrwith(wide.shift(lag)) for lag in [1, 48, 336]})
    lag_table.to_csv(reports / 'training_lag_correlations.csv')
    observed = panel.loc[panel.observed_row]
    frequencies = observed.groupby('household_id').timestamp.diff().value_counts().head(12)
    frequencies.rename_axis('time_delta').rename('count').to_csv(reports / 'observed_frequency.csv')
    summary = dict(training_slots=len(train), training_missing=int(train.energy_kwh.isna().sum()),
                   peak_hour=float(train.groupby('slot').energy_kwh.mean().idxmax()/2),
                   lowest_hour=float(train.groupby('slot').energy_kwh.mean().idxmin()/2),
                   weekday_mean_kwh=float(train.loc[~train.weekend, 'energy_kwh'].mean()),
                   weekend_mean_kwh=float(train.loc[train.weekend, 'energy_kwh'].mean()),
                   mean_kwh=float(train.energy_kwh.mean()), median_kwh=float(train.energy_kwh.median()),
                   p99_kwh=float(train.energy_kwh.quantile(.99)), max_kwh=float(train.energy_kwh.max()),
                   household_mean_min=float(train.groupby('household_id').energy_kwh.mean().min()),
                   household_mean_max=float(train.groupby('household_id').energy_kwh.mean().max()),
                   median_household_lag_correlation=lag_table.median().to_dict(),
                   monthly_complete_day_mean_kwh={str(k): v for k, v in monthly.items()})
    (reports / 'eda_summary.json').write_text(json.dumps(summary, indent=2))
    quality_by_split = []
    for name, start, end in [('train', '2012-09-01', '2013-09-01'), ('validation', '2013-09-01', '2013-11-01'), ('test', '2013-11-01', '2014-01-01')]:
        subset = panel.loc[panel.timestamp.ge(start) & panel.timestamp.lt(end)]
        quality_by_split.append(dict(split=name, slots=len(subset), missing=int(subset.energy_kwh.isna().sum()), coverage=float(subset.energy_kwh.notna().mean())))
    pd.DataFrame(quality_by_split).to_csv(reports / 'quality_by_split.csv', index=False)
    print(json.dumps(summary, indent=2))
    print(pd.DataFrame(quality_by_split).to_string(index=False))
    print(pd.read_csv(reports / 'household_quality.csv').sort_values('coverage').head().to_string(index=False))


if __name__ == '__main__':
    main()

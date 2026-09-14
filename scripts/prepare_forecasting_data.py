"""Prepare the selected households for forecasting experiments."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'archive'
OUT = ROOT / 'data' / 'processed'
REPORTS = ROOT / 'reports'

HISTORY_START = pd.Timestamp('2012-08-25')
TRAIN_START = pd.Timestamp('2012-09-01')
VALID_START = pd.Timestamp('2013-09-01')
END = pd.Timestamp('2014-01-01')


def longest_missing_run(values):
    missing = values.isna()
    return int(missing.groupby((~missing).cumsum()).sum().max())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    households = pd.read_csv(OUT / 'households.csv')
    selected_ids = set(households['LCLid'])
    parts = []

    for block in sorted(households['file'].unique()):
        path = RAW / 'halfhourly_dataset' / 'halfhourly_dataset' / f'{block}.csv'
        for chunk in pd.read_csv(path, chunksize=250_000, dtype=str, keep_default_na=False):
            keep_id = chunk['LCLid'].isin(selected_ids)
            if not keep_id.any():
                continue
            chunk = chunk.loc[keep_id].copy()
            chunk['timestamp'] = pd.to_datetime(chunk['tstp'], errors='coerce')
            keep_time = chunk['timestamp'].ge(HISTORY_START) & chunk['timestamp'].lt(END)
            parts.append(chunk.loc[keep_time, ['LCLid', 'timestamp', 'energy(kWh/hh)']])

    raw = pd.concat(parts, ignore_index=True)
    raw = raw.rename(columns={'LCLid': 'household_id', 'energy(kWh/hh)': 'raw_energy'})
    raw['energy_kwh'] = pd.to_numeric(raw['raw_energy'].str.strip(), errors='coerce')
    raw.loc[~np.isfinite(raw['energy_kwh']) | raw['energy_kwh'].lt(0), 'energy_kwh'] = np.nan
    keys = ['household_id', 'timestamp']
    raw = raw.loc[raw['timestamp'].notna() & raw['timestamp'].eq(raw['timestamp'].dt.floor('30min'))]
    conflicts = raw.groupby(keys)['energy_kwh'].nunique(dropna=False).gt(1)
    clean = raw.drop_duplicates(keys).set_index(keys).sort_index()
    clean['duplicate_conflict'] = conflicts
    clean.loc[clean['duplicate_conflict'], 'energy_kwh'] = np.nan
    clean['observed_row'] = True

    grid = pd.MultiIndex.from_product(
        [sorted(selected_ids), pd.date_range(HISTORY_START, END, freq='30min', inclusive='left')],
        names=keys,
    )
    panel = clean[['energy_kwh', 'observed_row', 'duplicate_conflict']].reindex(grid).reset_index()
    for column in ['observed_row', 'duplicate_conflict']:
        panel[column] = panel[column].astype('boolean').fillna(False).astype(bool)
    panel.to_parquet(OUT / 'forecasting_panel.parquet', index=False)

    training = panel.loc[panel['timestamp'].ge(TRAIN_START) & panel['timestamp'].lt(VALID_START)]
    rows = []
    for household, data in training.groupby('household_id'):
        rows.append({
            'household_id': household,
            'expected_slots': len(data),
            'valid_slots': data['energy_kwh'].notna().sum(),
            'coverage': data['energy_kwh'].notna().mean(),
            'longest_missing_slots': longest_missing_run(data['energy_kwh']),
            'longest_missing_hours': longest_missing_run(data['energy_kwh']) / 2,
        })
    coverage = pd.DataFrame(rows).merge(
        households, left_on='household_id', right_on='LCLid', how='left', validate='one_to_one'
    )
    coverage.to_csv(REPORTS / 'extended_training_coverage.csv', index=False)
    summary = {
        'training_start': str(TRAIN_START),
        'training_end_exclusive': str(VALID_START),
        'households': len(coverage),
        'households_at_least_99_percent': int(coverage['coverage'].ge(.99).sum()),
        'households_at_least_95_percent': int(coverage['coverage'].ge(.95).sum()),
        'minimum_coverage': float(coverage['coverage'].min()),
        'median_coverage': float(coverage['coverage'].median()),
        'maximum_missing_hours': float(coverage['longest_missing_hours'].max()),
    }
    (REPORTS / 'extended_training_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(coverage.sort_values('coverage')[['household_id', 'coverage', 'longest_missing_hours']].to_string(index=False))


if __name__ == '__main__':
    main()

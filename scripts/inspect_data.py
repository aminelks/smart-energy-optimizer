"""Inspect local files without reading all half-hourly consumption into memory."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'archive'
OUT = ROOT / 'reports'
OUT.mkdir(exist_ok=True)


def main():
    inventory = []
    for path in sorted(RAW.rglob('*')):
        if path.is_file():
            columns = list(pd.read_csv(path, nrows=0, encoding='latin-1').columns) if path.suffix == '.csv' else []
            inventory.append(dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size,
                                  format=path.suffix, columns=' | '.join(columns)))
    pd.DataFrame(inventory).to_csv(OUT / 'raw_file_inventory.csv', index=False)
    meta = pd.read_csv(RAW / 'informations_households.csv')
    print('HOUSEHOLDS', meta.shape, '\n', meta.groupby(['Acorn_grouped', 'stdorToU']).size())
    counts = meta.groupby(['Acorn_grouped', 'file', 'stdorToU']).size().unstack(fill_value=0)
    print('BLOCK CANDIDATES\n', counts.to_string())
    profiles = {}
    for name in ['informations_households.csv', 'weather_hourly_darksky.csv',
                 'weather_daily_darksky.csv', 'uk_bank_holidays.csv', 'acorn_details.csv']:
        frame = pd.read_csv(RAW / name, encoding='latin-1')
        profile = dict(rows=len(frame), columns=list(frame.columns),
                       missing=frame.isna().sum().to_dict(), duplicate_rows=int(frame.duplicated().sum()))
        date_col = 'time' if 'time' in frame else 'Bank holidays' if 'Bank holidays' in frame else None
        if date_col:
            dates = pd.to_datetime(frame[date_col], errors='coerce')
            profile.update(date_min=str(dates.min()), date_max=str(dates.max()),
                           invalid_dates=int(dates.isna().sum()), duplicate_dates=int(dates.duplicated().sum()),
                           common_deltas=dates.sort_values().diff().value_counts().head(4).astype(int).rename_axis('delta').reset_index().astype(str).to_dict('records'))
        profiles[name] = profile
    (OUT / 'table_profiles.json').write_text(json.dumps(profiles, indent=2))
    print('TABLE PROFILES\n', json.dumps(profiles, indent=2))
    daily_summary = []
    for path in sorted((RAW / 'daily_dataset' / 'daily_dataset').glob('*.csv')):
        frame = pd.read_csv(path, usecols=['LCLid', 'day', 'energy_count'])
        daily_summary.append(dict(file=path.stem, rows=len(frame), households=frame.LCLid.nunique(),
                                  date_min=frame.day.min(), date_max=frame.day.max(),
                                  incomplete_days=int(frame.energy_count.lt(48).sum()),
                                  overfull_days=int(frame.energy_count.gt(48).sum())))
    summary = pd.DataFrame(daily_summary)
    summary.to_csv(OUT / 'daily_block_profiles.csv', index=False)
    # The consolidated daily file is an alternative representation, not a second source to append.
    rows, first_day, last_day = 0, '9999', '0000'
    for chunk in pd.read_csv(RAW / 'daily_dataset.csv', usecols=['day'], chunksize=250_000):
        rows += len(chunk)
        first_day, last_day = min(first_day, chunk.day.min()), max(last_day, chunk.day.max())
    profiles['daily_dataset.csv'] = dict(rows=rows, date_min=first_day, date_max=last_day)
    for block in ['block_16', 'block_63', 'block_78']:
        path = RAW / 'hhblock_dataset' / 'hhblock_dataset' / f'{block}.csv'
        wide = pd.read_csv(path, usecols=['LCLid', 'day'])
        profiles[f'hhblock_dataset/{block}.csv'] = dict(rows=len(wide), households=wide.LCLid.nunique(),
                                                       date_min=wide.day.min(), date_max=wide.day.max())
    (OUT / 'table_profiles.json').write_text(json.dumps(profiles, indent=2))
    print('DAILY SUMMARY', summary.rows.sum(), summary.date_min.min(), summary.date_max.max())


if __name__ == '__main__':
    main()

"""Reusable versions of the preparation steps introduced in notebook 01."""
import numpy as np
import pandas as pd

START = pd.Timestamp('2012-09-01')
VALID_START = pd.Timestamp('2013-09-01')
TEST_START = pd.Timestamp('2013-11-01')
END = pd.Timestamp('2014-01-01')


def choose_blocks(metadata):
    """One block per known ACORN group, maximizing the smaller tariff group."""
    counts = metadata.groupby(['Acorn_grouped', 'file', 'stdorToU']).size().unstack(fill_value=0)
    chosen = []
    for group in ['Affluent', 'Comfortable', 'Adversity']:
        candidates = counts.loc[group].copy()
        candidates['balance'] = candidates[['Std', 'ToU']].min(axis=1)
        candidates['number'] = candidates.index.str.replace('block_', '').astype(int)
        chosen.append(candidates.sort_values(['balance', 'number'], ascending=[False, True]).index[0])
    return chosen


def read_blocks(raw_dir, blocks):
    """Stream three blocks; retain only the development period, not the full archive."""
    pieces, profiles = [], []
    for block in blocks:
        profile = dict(file=block, rows=0, invalid_timestamps=0, date_min=None, date_max=None)
        path = raw_dir / 'halfhourly_dataset' / 'halfhourly_dataset' / f'{block}.csv'
        for chunk in pd.read_csv(path, chunksize=250_000, dtype=str, keep_default_na=False):
            timestamp = pd.to_datetime(chunk.tstp, errors='coerce')
            profile['rows'] += len(chunk)
            profile['invalid_timestamps'] += int(timestamp.isna().sum())
            if timestamp.notna().any():
                low, high = str(timestamp.min()), str(timestamp.max())
                profile['date_min'] = min(profile['date_min'] or low, low)
                profile['date_max'] = max(profile['date_max'] or high, high)
            keep = timestamp.ge(START) & timestamp.lt(END)
            part = chunk.loc[keep].copy()
            part['timestamp'] = timestamp.loc[keep]
            pieces.append(part.rename(columns={'LCLid': 'household_id', 'energy(kWh/hh)': 'raw_energy'}).drop(columns='tstp'))
        profiles.append(profile)
    return pd.concat(pieces, ignore_index=True), pd.DataFrame(profiles)


def clean_observations(raw):
    """Keep high/zero readings. Invalidate negative/nonfinite values and conflicting pairs."""
    frame = raw.copy()
    numeric = pd.to_numeric(frame.raw_energy.str.strip(), errors='coerce')
    missing_token = frame.raw_energy.str.strip().str.lower().isin(['', 'null', 'nan', 'na'])
    finite = np.isfinite(numeric)
    negative = numeric.lt(0)
    frame['source_missing'] = missing_token
    frame['invalid_value'] = (~finite & ~missing_token) | negative
    frame['energy_kwh'] = numeric.where(finite & ~negative)
    keys = ['household_id', 'timestamp']
    off_grid = frame.timestamp.ne(frame.timestamp.dt.floor('30min'))
    reversed_steps = frame.groupby('household_id', sort=False).timestamp.diff().lt(pd.Timedelta(0))
    audit = dict(raw_rows=len(frame), missing_consumption=int(missing_token.sum()),
                 nonnumeric_nonmissing=int((numeric.isna() & ~missing_token).sum()),
                 negative_consumption=int(negative.sum()), nonfinite_consumption=int((numeric.notna() & ~finite).sum()),
                 invalid_timestamps=int(frame.timestamp.isna().sum()), off_grid_timestamps=int(off_grid.sum()),
                 exact_duplicate_rows=int(raw.duplicated().sum()),
                 duplicate_pairs=int(frame.duplicated(keys).sum()), reversed_steps=int(reversed_steps.sum()))
    frame = frame.loc[frame.timestamp.notna() & ~off_grid].copy()
    # Identical parsed values collapse; conflicting readings become missing, never averaged.
    conflicts = frame.groupby(keys).energy_kwh.nunique(dropna=False).gt(1)
    flags = frame.groupby(keys)[['source_missing', 'invalid_value']].max()
    frame = frame.drop_duplicates(keys).set_index(keys).sort_index()
    frame[['source_missing', 'invalid_value']] = flags
    frame['duplicate_conflict'] = conflicts
    frame.loc[frame.duplicate_conflict, 'energy_kwh'] = np.nan
    frame['observed_row'] = True
    audit['conflicting_pairs'] = int(conflicts.sum())
    return frame.drop(columns='raw_energy').reset_index(), audit


def regularize(frame, households, start=START, end=END):
    grid = pd.MultiIndex.from_product(
        [sorted(households), pd.date_range(start, end, freq='30min', inclusive='left')],
        names=['household_id', 'timestamp'])
    result = frame.set_index(['household_id', 'timestamp']).reindex(grid)
    for col in ['source_missing', 'invalid_value', 'duplicate_conflict', 'observed_row']:
        result[col] = result[col].astype('boolean').fillna(False).astype(bool)
    return result.reset_index()


def longest_missing_run(values):
    missing = values.isna()
    return int(missing.groupby((~missing).cumsum()).sum().max())


def household_quality(frame):
    grouped = frame.groupby('household_id')
    result = grouped.energy_kwh.agg(slots='size', valid='count', mean_kwh='mean', max_kwh='max')
    result['coverage'] = result.valid / result.slots
    result['missing_intervals'] = grouped.observed_row.apply(lambda s: int((~s).sum()))
    result['longest_missing_slots'] = grouped.energy_kwh.apply(longest_missing_run)
    result['longest_missing_hours'] = result.longest_missing_slots / 2
    return result.reset_index()


def select_households(training, metadata, per_stratum=4):
    """Select from training quality only; stable ID order within six metadata strata."""
    quality = household_quality(training).merge(metadata, left_on='household_id', right_on='LCLid', validate='one_to_one')
    quality['eligible'] = quality.coverage.ge(.99) & quality.longest_missing_slots.le(48)
    chosen = []
    for group in ['Affluent', 'Comfortable', 'Adversity']:
        for tariff in ['Std', 'ToU']:
            eligible = quality.loc[quality.eligible & quality.Acorn_grouped.eq(group) & quality.stdorToU.eq(tariff)]
            if len(eligible) < per_stratum:
                raise ValueError(f'Insufficient eligible households for {group}/{tariff}: {len(eligible)}')
            chosen.extend(eligible.sort_values('household_id').head(per_stratum).household_id)
    quality['selected'] = quality.household_id.isin(chosen)
    return sorted(chosen), quality

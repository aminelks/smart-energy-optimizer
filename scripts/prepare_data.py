"""Rebuild the deterministic development panel."""
from pathlib import Path
import json
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.preprocessing import (START, VALID_START, END, choose_blocks, read_blocks,
                                    clean_observations, regularize, select_households, household_quality)


def main():
    raw_dir = ROOT / 'archive'
    out, reports = ROOT / 'data' / 'processed', ROOT / 'reports'
    out.mkdir(parents=True, exist_ok=True)
    reports.mkdir(exist_ok=True)
    metadata = pd.read_csv(raw_dir / 'informations_households.csv')
    blocks = choose_blocks(metadata)
    print('Selected blocks:', blocks, flush=True)
    raw, profiles = read_blocks(raw_dir, blocks)
    profiles.to_csv(reports / 'halfhourly_block_profiles.csv', index=False)
    clean, _ = clean_observations(raw)
    candidates = regularize(clean.loc[clean.timestamp.lt(VALID_START)],
                            metadata.loc[metadata.file.isin(blocks), 'LCLid'].tolist(), end=VALID_START)
    ids, selection = select_households(candidates, metadata)
    selection.to_csv(reports / 'household_selection.csv', index=False)
    selected_raw = raw.loc[raw.household_id.isin(ids)]
    clean, audit = clean_observations(selected_raw)
    panel = regularize(clean, ids)
    # A training-only tail threshold marks unusual consumption; it never removes it.
    threshold = panel.loc[panel.timestamp.lt(VALID_START)].groupby('household_id').energy_kwh.quantile(.999)
    panel['high_usage_flag'] = panel.energy_kwh.gt(panel.household_id.map(threshold))
    threshold.rename('training_q999_kwh').to_csv(reports / 'high_usage_thresholds.csv')
    household_quality(panel).to_csv(reports / 'household_quality.csv', index=False)
    selected_meta = metadata.loc[metadata.LCLid.isin(ids)].sort_values('LCLid')
    selected_meta.to_csv(out / 'households.csv', index=False)
    panel.to_parquet(out / 'energy_halfhourly.parquet', index=False)
    audit.update(grid_rows=len(panel), households=len(ids), missing_intervals=int((~panel.observed_row).sum()),
                 missing_clean_consumption=int(panel.energy_kwh.isna().sum()),
                 zero_readings=int(panel.energy_kwh.eq(0).sum()), high_usage_flags=int(panel.high_usage_flag.sum()),
                 maximum_kwh=float(panel.energy_kwh.max()))
    (reports / 'quality_summary.json').write_text(json.dumps(audit, indent=2))
    manifest = dict(raw_directory='archive', blocks=blocks, start=str(START), end_exclusive=str(END),
                    selection_training_end_exclusive=str(VALID_START), households=ids,
                    selection='4 IDs per ACORN group x tariff; training coverage >=99%, maximum missing run <=24h; ID ascending',
                    timezone='Naive source timestamps; timezone/interval-label convention not established by local files')
    (out / 'selection_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(audit, indent=2))
    print(selection.groupby(['Acorn_grouped', 'stdorToU']).eligible.sum())


if __name__ == '__main__':
    main()

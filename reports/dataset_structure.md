# Local dataset structure

All 343 files were inventoried recursively in the existing `archive/` directory. The data was not downloaded, moved or edited.
`raw_file_inventory.csv` records each path, byte size, format and actual local CSV header.

| Family | Files / rows inspected | Size | Priority and use |
|---|---|---|---|
| `halfhourly_dataset/halfhourly_dataset` | 112 CSV headers; three full block scans | 7.325 GiB | Needed now: long half-hour consumption |
| `informations_households.csv` | 5,566 rows | See per-file inventory | Needed now: ID, tariff group, ACORN, block mapping |
| `daily_dataset/daily_dataset` | 112 CSVs; 3,510,433 household-day rows | 0.349 GiB | Optional: inspect coverage and daily aggregation |
| `daily_dataset.csv` | 3,510,433 rows | In the 0.330 GiB top-level files | Optional alternative daily representation; do not append to daily blocks |
| `hhblock_dataset/hhblock_dataset` | 112 headers; three representative ID/date scans | 1.556 GiB | Optional wide half-hour representation |
| `weather_hourly_darksky.csv` | 21,165 rows | See per-file inventory | Useful later: weather experiment during forecasting modeling |
| `weather_daily_darksky.csv` | 882 rows | See per-file inventory | Useful later, with timestamp alignment checks |
| `uk_bank_holidays.csv` | 25 rows | See per-file inventory | Useful later: optional calendar indicator |
| `acorn_details.csv` | Category-level descriptors | See per-file inventory | Optional context; Latin-1 encoding needed |
| `darksky_parameters_documentation.html` | Local API documentation | See per-file inventory | Useful later: field and units reference; does not prove export configuration |

No file is permanently ruled out. Wide and daily consumption alternatives are unnecessary for the exploration model panel. Weather is inspected but not joined. No interval price schedule was found.

## Actual schemas and coverage

- **Consumption:** `LCLid`, `tstp`, `energy(kWh/hh)`. The first two form the household/time key. Energy is in kWh per half-hour, not power in kW. The selected observed timestamps overwhelmingly advance by 30 minutes.
- **Household metadata:** `LCLid`, `stdorToU`, `Acorn`, `Acorn_grouped`, `file`. There are Std and ToU groups. These labels do not include prices or the dates when a tariff was applied.
- **Daily:** `LCLid`, `day`, `energy_median`, `energy_mean`, `energy_max`, `energy_count`, `energy_std`, `energy_sum`, `energy_min`. Both daily representations span 2011-11-23 to 2014-02-28. Equal row counts and schemas were checked; full content equivalence was not assumed.
- **Wide half-hour:** `LCLid`, `day`, `hh_0`–`hh_47`. Inspected blocks have 31,409, 32,352 and 30,615 household-day rows. Their ranges are recorded in `table_profiles.json`; the wide data is not assumed interchangeable at boundary days.
- **Hourly weather:** time, temperature, apparent temperature, dew point, humidity, pressure, wind, visibility, precipitation type and text conditions. Actual range: 2011-11-01 00:00 to 2014-03-31 22:00. No duplicate timestamps; two absent hours and 13 missing pressure readings.
- **Daily weather:** daily temperature extrema and their times, apparent temperatures, cloud cover, wind, pressure, humidity, UV, sunrise/sunset and text conditions. Actual range: 2011-11-01 00:00 to 2014-03-30 23:00. Some timestamps are 23:00, so a naive date join is unsafe.
- **ACORN details:** category reference rows and columns ACORN-A through ACORN-Q. These describe categories, not measured attributes for each household.

The exact scans of half-hourly candidate blocks found:

| Block | Households | Rows across all dates | First timestamp | Last timestamp |
|---|---:|---:|---|---|
| 16 | 50 | 1,519,233 | 2011-12-12 11:00 | 2014-02-28 00:00 |
| 63 | 50 | 1,562,825 | 2011-12-05 10:00 | 2014-02-28 00:00 |
| 78 | 50 | 1,480,814 | 2011-11-23 09:00 | 2014-02-28 00:00 |

The remaining half-hour blocks were not scanned for full row counts or exact date ranges. Full-source coverage is reported from the much smaller daily tables, with the three consumption scans providing direct confirmation for the candidate pool.

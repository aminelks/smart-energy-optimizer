"""Reusable version of the baseline calculations introduced in notebook 01."""
import numpy as np
import pandas as pd

BASELINES = {'Daily seasonal naive': 'lag_48', 'Weekly seasonal naive': 'lag_336'}


def evaluate_baselines(features):
    validation = features.loc[features['split'] == 'validation'].copy()
    common = validation.loc[validation['evaluation_eligible']]
    if common.empty:
        raise ValueError('No common validation targets can be scored.')
    overall, by_household, availability = [], [], []
    for name, column in BASELINES.items():
        error = common[column] - common['energy_kwh']
        mae = error.abs().mean()
        rmse = np.sqrt((error**2).mean())
        overall.append(dict(baseline=name, mae_kwh=mae,
                            rmse_kwh=rmse, scored_slots=len(common),
                            total_slots=len(validation), observed_targets=int(validation.energy_kwh.notna().sum()),
                            coverage_all_slots=len(common)/len(validation),
                            coverage_observed_targets=len(common)/validation.energy_kwh.notna().sum()))
        for household, group in common.assign(error=error).groupby('household_id'):
            by_household.append(dict(baseline=name, household_id=household, slots=len(group),
                                     mae_kwh=float(group.error.abs().mean()),
                                     rmse_kwh=float(np.sqrt((group.error**2).mean()))))
        available = validation.energy_kwh.notna() & validation[column].notna()
        availability.append(dict(baseline=name, predictions_available=int(validation[column].notna().sum()),
                                 independently_scorable_slots=int(available.sum()), common_scorable_slots=len(common)))
    return pd.DataFrame(overall), pd.DataFrame(by_household), pd.DataFrame(availability)

"""Run the controlled forecasting experiments and save reusable artifacts."""
from pathlib import Path
import json
import sys
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.features import make_forecasting_features, add_hourly_weather
from src.forecasting.train import fit_model
from src.forecasting.evaluate import regression_metrics
from src.forecasting.predict import add_predictions

TRAIN_START = pd.Timestamp('2012-09-01')
VALID_START = pd.Timestamp('2013-09-01')
TEST_START = pd.Timestamp('2013-11-01')
END = pd.Timestamp('2014-01-01')

HISTORY_FEATURES = ['lag_48', 'lag_336', 'previous_day_total', 'previous_week_total']
CALENDAR_FEATURES = ['slot', 'day_of_week', 'weekend', 'month']
WEATHER_SOURCE_COLUMNS = ['temperature', 'humidity', 'windSpeed', 'pressure']
WEATHER_FEATURES = [f'weather_{column}' for column in WEATHER_SOURCE_COLUMNS]


def score(name, group, data, prediction):
    metrics = regression_metrics(data['energy_kwh'].to_numpy(), prediction)
    metrics.update({'model': name, 'feature_groups': group})
    return metrics


def main():
    processed = ROOT / 'data' / 'processed'
    outputs = ROOT / 'outputs'
    models = ROOT / 'models'
    reports = ROOT / 'reports'
    outputs.mkdir(exist_ok=True)
    models.mkdir(exist_ok=True)

    panel = pd.read_parquet(processed / 'forecasting_panel.parquet')
    features = make_forecasting_features(panel, TRAIN_START, VALID_START, TEST_START, END)
    weather = pd.read_csv(ROOT / 'archive' / 'weather_hourly_darksky.csv', parse_dates=['time'])
    features = add_hourly_weather(features, weather, WEATHER_SOURCE_COLUMNS)
    features.to_parquet(processed / 'forecasting_model_features.parquet', index=False)

    training = features.loc[features['split'].eq('train') & features['energy_kwh'].notna()].copy()
    validation = features.loc[features['split'].eq('validation')].copy()
    validation_mask = validation['energy_kwh'].notna() & validation['lag_336'].notna()
    validation_common = validation.loc[validation_mask].copy()

    experiments = [
        ('Weekly naive', 'weekly lag', None),
        ('Model A', 'history', HISTORY_FEATURES),
        ('Model B', 'history + calendar', HISTORY_FEATURES + CALENDAR_FEATURES),
        ('Model C (weather oracle)', 'history + calendar + observed weather',
         HISTORY_FEATURES + CALENDAR_FEATURES + WEATHER_FEATURES),
    ]
    results = []
    experiment_models = {}
    for name, group, feature_names in experiments:
        if feature_names is None:
            prediction = validation_common['lag_336'].to_numpy()
        else:
            model = fit_model(training, feature_names)
            experiment_models[name] = model
            prediction = np.maximum(model.predict(validation_common[feature_names]), 0)
        results.append(score(name, group, validation_common, prediction))

    comparison = pd.DataFrame(results)
    baseline_mae = comparison.loc[comparison['model'].eq('Weekly naive'), 'mae_kwh'].iloc[0]
    model_a_mae = comparison.loc[comparison['model'].eq('Model A'), 'mae_kwh'].iloc[0]
    comparison['mae_gain_vs_weekly_percent'] = 100 * (baseline_mae - comparison['mae_kwh']) / baseline_mae
    comparison['mae_gain_vs_model_a_percent'] = 100 * (model_a_mae - comparison['mae_kwh']) / model_a_mae
    comparison.to_csv(outputs / 'forecasting_validation_comparison.csv', index=False)
    print('\nValidation experiments')
    print(comparison.to_string(index=False))

    realistic = comparison.loc[comparison['model'].isin(['Model A', 'Model B'])].sort_values('mae_kwh').iloc[0]
    if realistic['model'] == 'Model B':
        selected_features = HISTORY_FEATURES + CALENDAR_FEATURES
    else:
        selected_features = HISTORY_FEATURES

    tuning_options = [
        {'name': 'small trees', 'n_estimators': 350, 'learning_rate': 0.05, 'num_leaves': 15, 'min_child_samples': 100},
        {'name': 'default size', 'n_estimators': 400, 'learning_rate': 0.05, 'num_leaves': 31, 'min_child_samples': 50},
        {'name': 'slower learning', 'n_estimators': 700, 'learning_rate': 0.03, 'num_leaves': 31, 'min_child_samples': 80},
        {'name': 'larger trees', 'n_estimators': 500, 'learning_rate': 0.05, 'num_leaves': 63, 'min_child_samples': 100},
    ]
    tuning_rows = []
    tuned_models = {}
    for option in tuning_options:
        params = {key: value for key, value in option.items() if key != 'name'}
        model = fit_model(training, selected_features, params)
        prediction = np.maximum(model.predict(validation_common[selected_features]), 0)
        metrics = score(option['name'], realistic['feature_groups'], validation_common, prediction)
        metrics.update(params)
        tuning_rows.append(metrics)
        tuned_models[option['name']] = model
    tuning = pd.DataFrame(tuning_rows).sort_values('mae_kwh').reset_index(drop=True)
    tuning.to_csv(outputs / 'forecasting_tuning.csv', index=False)
    print('\nSmall manual tuning')
    print(tuning.to_string(index=False))

    best_name = tuning.loc[0, 'model']
    best_params = {
        key: tuning.loc[0, key].item() if hasattr(tuning.loc[0, key], 'item') else tuning.loc[0, key]
        for key in ['n_estimators', 'learning_rate', 'num_leaves', 'min_child_samples']
    }
    best_model = tuned_models[best_name]
    validation_predictions = add_predictions(validation_common, best_model, selected_features)

    households = pd.read_csv(processed / 'households.csv').rename(columns={'LCLid': 'household_id'})
    analysis = validation_predictions.merge(
        households[['household_id', 'stdorToU', 'Acorn_grouped']], on='household_id', how='left', validate='many_to_one'
    )
    by_household = analysis.groupby('household_id').agg(
        rows=('absolute_error', 'size'), mae_kwh=('absolute_error', 'mean'), rmse_kwh=('residual', lambda x: np.sqrt(np.mean(x ** 2)))
    ).reset_index()
    by_slot = analysis.groupby('slot').agg(mae_kwh=('absolute_error', 'mean'), rows=('absolute_error', 'size')).reset_index()
    by_day = analysis.assign(day=analysis['timestamp'].dt.floor('D')).groupby('day').agg(
        mae_kwh=('absolute_error', 'mean'), rows=('absolute_error', 'size')
    ).reset_index()
    by_weekend = analysis.groupby('weekend').agg(mae_kwh=('absolute_error', 'mean'), rows=('absolute_error', 'size')).reset_index()
    by_tariff = analysis.groupby('stdorToU').agg(mae_kwh=('absolute_error', 'mean'), rows=('absolute_error', 'size')).reset_index()
    by_acorn = analysis.groupby('Acorn_grouped').agg(mae_kwh=('absolute_error', 'mean'), rows=('absolute_error', 'size')).reset_index()
    by_household.to_csv(outputs / 'forecasting_error_by_household.csv', index=False)
    by_slot.to_csv(outputs / 'forecasting_error_by_slot.csv', index=False)
    by_day.to_csv(outputs / 'forecasting_error_by_day.csv', index=False)
    by_weekend.to_csv(outputs / 'forecasting_error_by_weekend.csv', index=False)
    by_tariff.to_csv(outputs / 'forecasting_error_by_tariff.csv', index=False)
    by_acorn.to_csv(outputs / 'forecasting_error_by_acorn.csv', index=False)
    analysis.nlargest(20, 'absolute_error')[
        ['household_id', 'timestamp', 'energy_kwh', 'predicted_kwh', 'residual', 'absolute_error']
    ].to_csv(outputs / 'forecasting_largest_errors.csv', index=False)

    # The model and tuning choices are now fixed. Refit on train + validation, then score test once.
    final_training = features.loc[features['split'].isin(['train', 'validation']) & features['energy_kwh'].notna()].copy()
    final_model = fit_model(final_training, selected_features, best_params)
    test = features.loc[features['split'].eq('test')].copy()
    test_mask = test['energy_kwh'].notna() & test['lag_336'].notna()
    test_common = test.loc[test_mask].copy()
    test_predictions = add_predictions(test_common, final_model, selected_features)
    test_model_metrics = regression_metrics(test_predictions['energy_kwh'].to_numpy(), test_predictions['predicted_kwh'].to_numpy())
    test_weekly_metrics = regression_metrics(test_common['energy_kwh'].to_numpy(), test_common['lag_336'].to_numpy())
    test_gain = 100 * (test_weekly_metrics['mae_kwh'] - test_model_metrics['mae_kwh']) / test_weekly_metrics['mae_kwh']

    saved_predictions = pd.concat([
        validation_predictions.assign(split='validation'),
        test_predictions.assign(split='test'),
    ], ignore_index=True)
    prediction_columns = [
        'household_id', 'timestamp', 'forecast_origin', 'split', 'energy_kwh',
        'predicted_kwh', 'residual', 'absolute_error'
    ]
    saved_predictions[prediction_columns].to_parquet(processed / 'forecasting_predictions.parquet', index=False)
    joblib.dump(final_model, models / 'forecasting_model.joblib')

    selected = {
        'model_family': 'LightGBM LGBMRegressor',
        'selected_experiment': str(realistic['model']),
        'features': selected_features,
        'parameters': best_params,
        'training_period_for_final_model': '2012-09-01 to 2013-10-31',
        'forecast_origin': 'midnight before the 48 target slots',
        'weather_oracle_is_operational': False,
    }
    (models / 'forecasting_model_metadata.json').write_text(json.dumps(selected, indent=2))
    metrics = {
        'validation_common_rows': int(len(validation_common)),
        'validation_experiments': comparison.to_dict('records'),
        'selected_validation_model': str(realistic['model']),
        'selected_tuning_option': best_name,
        'selected_validation_metrics': regression_metrics(
            validation_predictions['energy_kwh'].to_numpy(), validation_predictions['predicted_kwh'].to_numpy()
        ),
        'test_common_rows': int(len(test_common)),
        'test_selected_model': test_model_metrics,
        'test_weekly_naive': test_weekly_metrics,
        'test_mae_gain_vs_weekly_percent': float(test_gain),
    }
    (outputs / 'forecasting_metrics.json').write_text(json.dumps(metrics, indent=2))
    print('\nFinal test')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()

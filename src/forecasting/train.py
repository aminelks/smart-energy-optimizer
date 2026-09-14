"""Fit one LightGBM model with a small parameter dictionary."""
from lightgbm import LGBMRegressor


DEFAULT_PARAMS = {
    'objective': 'regression_l1',
    'n_estimators': 400,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_child_samples': 50,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': -1,
}


def fit_model(data, feature_names, params=None):
    settings = DEFAULT_PARAMS.copy()
    if params:
        settings.update(params)
    model = LGBMRegressor(**settings)
    model.fit(data[feature_names], data['energy_kwh'])
    return model

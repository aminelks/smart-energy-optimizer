"""Regression metrics used for all forecasting comparisons."""
import numpy as np


def regression_metrics(actual, predicted):
    error = predicted - actual
    return {
        'mae_kwh': float(np.abs(error).mean()),
        'rmse_kwh': float(np.sqrt(np.mean(error ** 2))),
        'rows': int(len(actual)),
    }


def score_model(data, prediction_column):
    return regression_metrics(data['energy_kwh'].to_numpy(), data[prediction_column].to_numpy())

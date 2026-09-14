"""Create a prediction table with errors for later analysis."""
import numpy as np


def add_predictions(data, model, feature_names):
    result = data.copy()
    result['predicted_kwh'] = np.maximum(model.predict(result[feature_names]), 0)
    result['residual'] = result['energy_kwh'] - result['predicted_kwh']
    result['absolute_error'] = result['residual'].abs()
    return result

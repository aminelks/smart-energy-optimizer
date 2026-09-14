import unittest
import numpy as np
import pandas as pd

from src.data.features import make_forecasting_features
from src.forecasting.evaluate import regression_metrics


class ForecastingTests(unittest.TestCase):
    def make_panel(self):
        timestamps = pd.date_range('2012-08-25', '2012-09-10', freq='30min', inclusive='left')
        return pd.DataFrame({
            'household_id': 'A',
            'timestamp': timestamps,
            'energy_kwh': np.arange(len(timestamps), dtype=float),
        })

    def test_features_use_only_earlier_days(self):
        panel = self.make_panel()
        origin = pd.Timestamp('2012-09-05')
        before = make_forecasting_features(
            panel, pd.Timestamp('2012-09-01'), pd.Timestamp('2012-09-07'),
            pd.Timestamp('2012-09-09'), pd.Timestamp('2012-09-10')
        )
        changed = panel.copy()
        changed.loc[changed['timestamp'] >= origin, 'energy_kwh'] = -999
        after = make_forecasting_features(
            changed, pd.Timestamp('2012-09-01'), pd.Timestamp('2012-09-07'),
            pd.Timestamp('2012-09-09'), pd.Timestamp('2012-09-10')
        )
        columns = ['lag_48', 'lag_336', 'previous_day_total', 'previous_week_total']
        left = before.loc[before['forecast_origin'] == origin, columns].reset_index(drop=True)
        right = after.loc[after['forecast_origin'] == origin, columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(left, right)

    def test_lag_keeps_exact_timestamp_gap(self):
        panel = self.make_panel()
        missing_time = pd.Timestamp('2012-09-01 12:00')
        panel.loc[panel['timestamp'] == missing_time, 'energy_kwh'] = np.nan
        features = make_forecasting_features(
            panel, pd.Timestamp('2012-09-01'), pd.Timestamp('2012-09-07'),
            pd.Timestamp('2012-09-09'), pd.Timestamp('2012-09-10')
        )
        target = features.loc[features['timestamp'] == missing_time + pd.Timedelta(days=1)].iloc[0]
        self.assertTrue(np.isnan(target['lag_48']))

    def test_metrics_match_hand_calculation(self):
        actual = np.array([1.0, 3.0])
        predicted = np.array([2.0, 1.0])
        metrics = regression_metrics(actual, predicted)
        self.assertEqual(metrics['mae_kwh'], 1.5)
        self.assertAlmostEqual(metrics['rmse_kwh'], np.sqrt(2.5))
        self.assertEqual(metrics['rows'], 2)


if __name__ == '__main__':
    unittest.main()

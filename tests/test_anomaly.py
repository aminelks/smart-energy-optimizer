import unittest

import pandas as pd

from src.anomaly.detector import fit_residual_calibration, score_residuals, make_episodes


class AnomalyTests(unittest.TestCase):
    def test_calibration_uses_given_past_frame(self):
        validation = pd.DataFrame({
            'household_id': ['A'] * 5,
            'residual': [-2.0, -1.0, 0.0, 1.0, 2.0],
        })
        calibration = fit_residual_calibration(validation, minimum_rows=3)
        future = pd.DataFrame({'household_id': ['A'], 'residual': [100.0]})
        scored = score_residuals(future, calibration)
        self.assertEqual(calibration.loc[0, 'residual_median'], 0.0)
        self.assertGreater(scored.loc[0, 'anomaly_score'], 60)

    def test_consecutive_flags_form_one_episode(self):
        frame = pd.DataFrame({
            'household_id': ['A', 'A', 'A'],
            'timestamp': pd.to_datetime(['2013-11-01 10:00', '2013-11-01 10:30', '2013-11-01 11:30']),
            'split': ['test'] * 3,
            'is_anomaly': [True] * 3,
            'anomaly_score': [5.0, 6.0, 7.0],
            'absolute_error': [1.0, 2.0, 3.0],
            'anomaly_direction': ['higher_than_expected'] * 3,
        })
        episodes = make_episodes(frame)
        self.assertEqual(sorted(episodes['duration_slots']), [1, 2])


if __name__ == '__main__':
    unittest.main()

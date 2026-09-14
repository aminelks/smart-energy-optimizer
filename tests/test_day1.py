"""Small regression checks for leakage, time alignment and data-quality policies."""
import unittest
import numpy as np
import pandas as pd
from src.data.features import make_day_ahead_features, HISTORY_FEATURES, CALENDAR_FEATURES
from src.data.preprocessing import clean_observations, regularize
from src.evaluation import evaluate_baselines


class Day1Tests(unittest.TestCase):
    def panel(self):
        dates = pd.date_range('2013-08-20', '2013-09-05', freq='30min', inclusive='left')
        return pd.concat([pd.DataFrame({'household_id': household, 'timestamp': dates,
                                        'energy_kwh': offset + np.arange(len(dates), dtype=float)})
                          for household, offset in [('A', 0), ('B', 10_000)]], ignore_index=True)

    def test_forecast_origin_prevents_leakage(self):
        panel = self.panel()
        origin = pd.Timestamp('2013-09-02')
        before = make_day_ahead_features(panel)
        panel.loc[panel.timestamp.ge(origin), 'energy_kwh'] = -999_999
        after = make_day_ahead_features(panel)
        cols = HISTORY_FEATURES + CALENDAR_FEATURES
        pd.testing.assert_frame_equal(before.loc[before.forecast_origin.eq(origin), cols],
                                      after.loc[after.forecast_origin.eq(origin), cols])
        self.assertEqual(before.loc[before.forecast_origin.eq(origin), 'slot'].nunique(), 48)

    def test_timestamp_lookup_does_not_jump_over_gaps_or_households(self):
        panel = self.panel()
        missing = pd.Timestamp('2013-09-01 12:00')
        panel = panel.loc[~(panel.household_id.eq('A') & panel.timestamp.eq(missing))]
        result = make_day_ahead_features(panel)
        target = result.loc[result.timestamp.eq(missing + pd.Timedelta(days=1))].set_index('household_id')
        self.assertTrue(np.isnan(target.loc['A', 'lag_48']))
        self.assertGreater(target.loc['B', 'lag_48'], 10_000)

    def test_cleaning_keeps_spikes_and_flags_conflicts(self):
        raw = pd.DataFrame({'household_id': ['A']*7,
                            'timestamp': pd.to_datetime(['2013-01-01 00:00', '2013-01-01 00:00',
                                                         '2013-01-01 00:30', '2013-01-01 00:30',
                                                         '2013-01-01 01:00', '2013-01-01 01:30', '2013-01-01 02:00']),
                            'raw_energy': ['0', '0', '1', '2', '-1', 'Null', '100']})
        clean, audit = clean_observations(raw)
        self.assertEqual(audit['exact_duplicate_rows'], 1)
        self.assertEqual(audit['conflicting_pairs'], 1)
        self.assertEqual(audit['negative_consumption'], 1)
        self.assertEqual(clean.energy_kwh.max(), 100)
        self.assertEqual(clean.energy_kwh.isna().sum(), 3)
        panel = regularize(clean, ['A'], start=pd.Timestamp('2013-01-01'), end=pd.Timestamp('2013-01-02'))
        self.assertEqual(len(panel), 48)
        self.assertFalse(panel.iloc[-1].observed_row)
        self.assertTrue(pd.isna(panel.iloc[-1].energy_kwh))

    def test_split_boundaries_and_common_metrics(self):
        frame = pd.DataFrame({'household_id': ['A']*3,
                              'timestamp': pd.to_datetime(['2013-08-31 23:30', '2013-09-01', '2013-11-01'], format='mixed'),
                              'energy_kwh': [1., 2., 3.]})
        self.assertEqual(make_day_ahead_features(frame)['split'].tolist(), ['train', 'validation', 'test'])
        sample = pd.DataFrame({'household_id': ['A']*4, 'split': ['validation']*3+['test'],
                               'energy_kwh': [1., 3., 5., 1e9], 'lag_48': [2., 1., np.nan, 0.],
                               'lag_336': [1., 5., 5., 0.], 'evaluation_eligible': [True, True, False, True]})
        scores, _, _ = evaluate_baselines(sample)
        self.assertEqual(scores.scored_slots.tolist(), [2, 2])
        self.assertEqual(scores.mae_kwh.tolist(), [1.5, 1.0])


if __name__ == '__main__':
    unittest.main()

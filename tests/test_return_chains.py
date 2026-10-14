"""返回链行为与反例验证；合成波形不提供实验准确率。"""
import unittest

import numpy as np
import pandas as pd

from modules.局部事件模型 import LocalEventConfig, fit_local_events
from modules.局部事件链模型 import EVENT_MODEL_VERSION, fit_local_return_chains
from modules.放电状态识别 import extract_features


def shaped_current(nodes, levels, *, gain=1., sign=1., shift=0., voltage=None):
    points = np.arange(2145)
    current = sign*gain*8*np.round(np.interp(points, nodes, levels)/8)+shift
    if voltage is None:
        voltage = np.where(points < 270, 20., 14.)
    return pd.DataFrame({'time': points/1e6, 'voltage': voltage, 'current': current})


def split_return(**options):
    return shaped_current([0, 350, 850, 920, 960, 1080, 1150, 2144],
                          [0, 0, 120, 120, 104, 40, 0, 0], **options)


def evaluate(data, budget=10):
    return fit_local_return_chains(data, extract_features(data),
                                   LocalEventConfig(max_segments=budget))


class ReturnChainTests(unittest.TestCase):
    def test_split_return_uses_observed_compatible_background_after_both_legs(self):
        data = split_return()
        result = evaluate(data)
        self.assertTrue(result['event_supported'])
        self.assertGreaterEqual(result['event_return_chain_segments'], 2)
        self.assertLess(result['event_return_chain_start_index'], 1000)
        self.assertGreater(result['event_return_chain_end_index'], 1100)
        self.assertGreaterEqual(result['event_post_reference_start_index'],
                                result['event_return_chain_end_index'])
        for role in ('pre', 'post', 'peak', 'voltage'):
            self.assertEqual(result[f'event_{role}_reference_end_index']-
                             result[f'event_{role}_reference_start_index'], 32)
        self.assertLess(result['event_post_reference_start_index'], 1200)
        self.assertGreater(result['event_bic_gain'], 0)
        self.assertGreaterEqual(result['event_joint_response_samples'], 32)

    def test_gain_polarity_and_large_dc_translation_do_not_determine_support(self):
        for gain in (.01, 1., 10.):
            for sign in (-1., 1.):
                with self.subTest(gain=gain, sign=sign):
                    result = evaluate(split_return(gain=gain, sign=sign, shift=1e5))
                    self.assertTrue(result['event_supported'])
                    self.assertEqual(result['event_orientation'], sign)

    def test_many_small_steps_cannot_hide_a_resolved_reverse_leg(self):
        points = np.arange(2145)
        # Each rebound boundary is one 8 A grid step, below 4*sigma; the complete
        # 40 A rebound is resolved and separates the two current excursions.
        data = shaped_current([0, 350, 700, 950, 1200, 1450, 1750, 2144],
                              [0, 0, 80, 40, 80, 80, 0, 0],
                              voltage=np.where((points >= 350) & (points < 950), 10., 20.))
        for budget in (4, 6, 8, 10):
            with self.subTest(budget=budget):
                self.assertFalse(evaluate(data, budget)['event_supported'])

    def test_two_events_do_not_borrow_voltage_from_the_first(self):
        points = np.arange(2145)
        current = np.zeros(2145)
        current[350:700] = 8*np.round(np.linspace(0, 40, 350)/8)
        current[1050:1350] = 8*np.round(np.linspace(0, 80, 300)/8)
        current[1350:1700] = 8*np.round(np.linspace(80, 0, 350)/8)
        for second_responds in (False, True):
            voltage = np.full(2145, 20.)
            voltage[350:700] = 10.
            if second_responds:
                voltage[1050:1700] = 10.
            data = pd.DataFrame({'time': points/1e6, 'voltage': voltage, 'current': current})
            result = evaluate(data)
            with self.subTest(second_responds=second_responds):
                self.assertEqual(result['event_supported'], second_responds)
                if second_responds:
                    self.assertGreaterEqual(result['event_start_index'], 1050)
                    self.assertGreaterEqual(result['event_voltage_reference_start_index'], 700)

    def test_valley_between_positive_pulses_is_not_an_independent_negative_event(self):
        points = np.arange(2145)
        data = shaped_current([0, 200, 400, 800, 900, 1050, 1150, 1500, 1700, 2144],
                              [0, 0, 80, 80, 0, 0, 80, 80, 0, 0],
                              # Voltage changes only while current is on its zero
                              # valley, not while either positive pulse is active.
                              voltage=np.where((points >= 930) & (points < 1050), 5., 20.))
        for budget in (4, 6, 8, 10):
            with self.subTest(budget=budget):
                self.assertFalse(evaluate(data, budget)['event_supported'])

    def test_peak_plateaus_cannot_be_used_as_the_valley_background(self):
        points = np.arange(2145)
        data = shaped_current([0, 200, 400, 599, 600, 800, 900, 1050,
                               1150, 1500, 1700, 2144],
                              [0, 0, 64, 64, 80, 80, 0, 0, 80, 80, 0, 0],
                              voltage=np.where((points >= 930) & (points < 1050), 5., 20.))
        features = extract_features(data)
        config = LocalEventConfig(max_segments=10)
        legacy = fit_local_events(data, features, config)
        # Retain legacy output for replay, but isolate its negative valley proposal
        # in v2: the two 80 A references are themselves positive pulse plateaus.
        self.assertTrue(legacy['event_supported'])
        self.assertEqual(legacy['event_orientation'], -1)
        result = fit_local_return_chains(data, features, config)
        self.assertFalse(result['event_supported'])
        self.assertIn('ambiguous_background_owned_by_opposite_excursions',
                      result['event_candidate_rejection_reasons'])

    def test_constant_or_later_voltage_does_not_support_a_return_chain(self):
        for late in (False, True):
            data = split_return()
            data['voltage'] = 20.
            if late:
                data.loc[1600:, 'voltage'] = 3.
            with self.subTest(late=late):
                self.assertFalse(evaluate(data)['event_supported'])

    def test_short_prior_voltage_spike_is_not_a_drive_reference(self):
        data = split_return()
        data['voltage'] = 20.
        data.loc[:19, 'voltage'] = 40.
        self.assertFalse(evaluate(data)['event_supported'])

    def test_steps_drift_and_sparse_narrow_peaks_do_not_gain_return_support(self):
        points = np.arange(2145)
        for pattern in ('step', 'drift', 'sparse', 'one_grid'):
            data = split_return()
            if pattern == 'step':
                current = np.where(points < 600, 0., -40.)
                current += np.random.default_rng(7).normal(0, .05, len(points))
            elif pattern == 'drift':
                current = np.linspace(-25., -40., len(points))
                current += np.random.default_rng(7).normal(0, .05, len(points))
            else:
                current = np.zeros(len(points))
                if pattern == 'one_grid':
                    current[700:1200] = 8.
                    current[50] = -8.
                else:
                    current[30] = 8.
                    for k, first in enumerate(range(200, 1900, 65)):
                        current[first:first+5] = (-1 if k % 2 else 1)*320.
            data['current'] = current
            with self.subTest(pattern=pattern):
                self.assertFalse(evaluate(data)['event_supported'])

    def test_untrusted_scale_cannot_be_overridden_by_return_shape(self):
        data = split_return()
        features = extract_features(data)
        for reliable in (False, np.nan, 'True'):
            features['current_scale_reliable'] = reliable
            with self.subTest(reliable=reliable):
                result = fit_local_return_chains(data, features)
                self.assertFalse(result['event_supported'])
                self.assertEqual(result['event_reason'], 'untrusted_current_scale')

    def test_v1_output_and_input_are_preserved_after_running_v2(self):
        data = split_return()
        features = extract_features(data)
        original_data = data.copy(deep=True)
        original_features = pd.Series(features).copy(deep=True)
        config = LocalEventConfig(max_segments=10)
        before = fit_local_events(data, features, config)
        result = fit_local_return_chains(data, features, config)
        after = fit_local_events(data, features, config)
        self.assertEqual(before, after)
        self.assertEqual(before['event_model_version'], 'local-piecewise-event-v1')
        self.assertFalse(any(name.startswith('event_return_chain_') for name in before))
        self.assertEqual(result['event_model_version'], EVENT_MODEL_VERSION)
        self.assertFalse(result['event_return_chain_changes_original_state'])
        pd.testing.assert_frame_equal(data, original_data)
        pd.testing.assert_series_equal(pd.Series(features), original_features)


if __name__ == '__main__':
    unittest.main()

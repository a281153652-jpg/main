"""局部事件模型的行为/反例验证；合成波形不代表实验准确率。"""
import unittest

import numpy as np
import pandas as pd

from modules.局部事件模型 import LocalEventConfig, fit_local_events
from modules.放电状态识别 import DynamicConfig, extract_features


def pulse(width=350, *, gain=1., sign=1., shift=0.):
    n, start, fall = 2145, 680, 64
    q, pre, post = 7.97869682312011, -1.363, -17.320
    end = start + width
    levels = np.interp(np.arange(n), [0, start, end-fall, end, n-1],
                       [pre, pre, 70.45, post, post])
    current = sign*gain*(pre+q*np.round((levels-pre)/q)) + shift
    voltage = np.where(np.arange(n) < start-80, 20., 14.)
    voltage += np.random.default_rng(40).normal(0, .02, n)
    return pd.DataFrame({'time': np.arange(n)/1e6, 'voltage': voltage, 'current': current})


def evidence(data, config=None):
    return fit_local_events(data, extract_features(data, config))


class LocalEventTests(unittest.TestCase):
    def test_event_duration_is_fitted_without_cycle_fraction_requirement(self):
        for width in (180, 350, 500, 650):
            with self.subTest(width=width):
                result = evidence(pulse(width))
                self.assertTrue(result['event_supported'])
                self.assertGreater(result['event_bic_gain'], 0)
                self.assertGreaterEqual(result['event_joint_response_samples'], 32)
        self.assertLess(180/2145, .25)

    def test_quantized_stair_steps_at_rise_start_are_not_made_background(self):
        result = evidence(pulse(500))
        # 前几个近常值台阶仍属于爬升，参考必须来自早于最初爬升的背景。
        self.assertTrue(result['event_supported'])
        self.assertLess(result['event_pre_reference_end_index'], 730)

    def test_gain_and_polarity_preserve_joint_event_support(self):
        for gain in (.01, 1., 10.):
            for sign in (-1., 1.):
                with self.subTest(gain=gain, sign=sign):
                    self.assertTrue(evidence(pulse(gain=gain, sign=sign))['event_supported'])

    def test_large_dc_translation_is_shape_only_without_zero_calibration(self):
        data = pulse(shift=1e6)
        original = data.copy(deep=True)
        features = extract_features(data)
        saved = pd.Series(features).copy(deep=True)
        result = fit_local_events(data, features)
        self.assertTrue(result['event_supported'])
        self.assertEqual(features['current_zero_offset_A'], 0.)
        pd.testing.assert_frame_equal(data, original)
        pd.testing.assert_series_equal(pd.Series(features), saved)

    def test_explicit_measured_offset_is_applied_once_for_shape(self):
        self.assertTrue(evidence(pulse(shift=5000),
                                 DynamicConfig(current_zero_offset=5000))['event_supported'])

    def test_constant_voltage_and_voltage_change_after_event_do_not_support(self):
        for late in (False, True):
            data = pulse()
            data['voltage'] = 20.
            if late:
                data.loc[1200:, 'voltage'] = 3.
            with self.subTest(late=late):
                self.assertFalse(evidence(data)['event_supported'])

    def test_short_prior_voltage_transient_is_not_a_stable_drive_reference(self):
        data = pulse()
        data['voltage'] = 20.
        data.loc[:19, 'voltage'] = 40.
        self.assertFalse(evidence(data)['event_supported'])

    def test_reverse_boundary_does_not_join_events_or_borrow_voltage_response(self):
        n, q = 2145, 8.
        current = np.zeros(n)
        current[350:700] = q*np.round(np.linspace(0, 40, 350)/q)
        current[1050:1350] = q*np.round(np.linspace(0, 80, 300)/q)
        current[1350:1700] = q*np.round(np.linspace(80, 0, 350)/q)
        for second_responds in (False, True):
            voltage = np.full(n, 20.)
            voltage[350:700] = 10.
            if second_responds:
                voltage[1050:1700] = 10.
            data = pd.DataFrame({'time': np.arange(n)/1e6, 'voltage': voltage, 'current': current})
            with self.subTest(second_responds=second_responds):
                result = evidence(data)
                self.assertEqual(result['event_supported'], second_responds)
                if second_responds:
                    self.assertGreaterEqual(result['event_start_index'], 1050)
                    self.assertGreaterEqual(result['event_pre_reference_start_index'], 700)

    def test_prior_voltage_ripple_uses_observed_reference_variation(self):
        data = pulse()
        data['voltage'] += .15*np.cos(2*np.pi*np.arange(len(data))/20)
        result = evidence(data)
        self.assertTrue(result['event_supported'])
        self.assertGreaterEqual(result['event_voltage_response_scale_V'],
                                result['event_voltage_reference_std_V'])

    def test_one_grid_plateau_and_sparse_multilevel_spikes_are_not_resolved_pulses(self):
        n, q = 2145, 7.97869682312011
        for pattern in ('one_grid', 'dense_grid', 'large_spikes'):
            data = pulse()
            current = np.full(n, -1.363)
            if pattern == 'one_grid':
                current[700:1200] += q
                current[50] -= q  # 第三电平能估计网格，但幅度仍不足。
            else:
                for k, first in enumerate(range(200, 1900, 65)):
                    length = 7 if pattern == 'dense_grid' else 5
                    current[first:first+length] += (-1 if k % 2 else 1)*q*(1 if pattern == 'dense_grid' else 40)
                if pattern == 'large_spikes':
                    current[30] += q  # 保持可分辨的小网格，不能靠大噪声尺度拒绝反例。
            data['current'] = current
            with self.subTest(pattern=pattern):
                self.assertFalse(evidence(data)['event_supported'])

    def test_single_step_and_monotonic_drift_have_no_complete_return(self):
        for kind in ('step', 'drift'):
            data = pulse()
            data['current'] = (np.where(np.arange(len(data)) < 600, 0., -40.)
                               if kind == 'step' else np.linspace(-25., -40., len(data)))
            data['current'] += np.random.default_rng(7).normal(0, .05, len(data))
            self.assertTrue(extract_features(data)['current_scale_reliable'])
            with self.subTest(kind=kind):
                self.assertFalse(evidence(data)['event_supported'])

    def test_truncated_incoming_or_outgoing_pulse_is_not_invented_complete(self):
        for incoming in (True, False):
            data = pulse()
            q = 7.97869682312011
            level = np.linspace(70., -17., len(data)) if incoming else np.linspace(-17., 70., len(data))
            data['current'] = -1.363+q*np.round((level+1.363)/q)
            with self.subTest(incoming=incoming):
                self.assertFalse(evidence(data)['event_supported'])

    def test_incompatible_backgrounds_require_further_context(self):
        data = pulse()
        data.loc[:680, 'current'] += 10*7.97869682312011
        self.assertFalse(evidence(data)['event_supported'])

    def test_unknown_noise_scale_cannot_be_promoted_by_structure(self):
        data = pulse()
        features = extract_features(data)
        for reliable in (False, np.nan, 'True'):
            features['current_scale_reliable'] = reliable
            with self.subTest(reliable=reliable):
                result = fit_local_events(data, features)
                self.assertFalse(result['event_supported'])
                self.assertEqual(result['event_reason'], 'untrusted_current_scale')

    def test_invalid_timebase_and_inconsistent_feature_count_are_rejected(self):
        data = pulse()
        features = extract_features(data)
        data.loc[1, 'time'] = data.loc[0, 'time']
        self.assertEqual(fit_local_events(data, features)['event_reason'], 'invalid_local_timebase')
        data = pulse()
        features['sample_count'] -= 1
        self.assertFalse(fit_local_events(data, features)['event_supported'])

    def test_integer_configuration_and_finite_positive_scale_are_required(self):
        for values in ({'min_segment_samples': 32.5}, {'max_segments': True},
                       {'max_boundary_candidates': 7}, {'noise_multiplier': np.nan},
                       {'noise_multiplier': True}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                LocalEventConfig(**values)


if __name__ == '__main__':
    unittest.main()

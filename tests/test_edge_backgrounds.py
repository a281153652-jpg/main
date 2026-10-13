"""不同前后背景下的完整脉冲，不把四个开发样本当成独立准确率验证。"""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

from modules.放电状态识别 import extract_features, classify_with_evidence, MODEL_VERSION
from modules.批次自学习 import learn_batch_states
from test_batch_learning import batch_features
from test_discharge_states import waveform


def shifted_background_pulse(scale=1., polarity=1., start_level=0.):
    n = 2145
    x = np.arange(n) / n
    q = 7.97869682312011
    pre, post = -1.36315727233887 + start_level * q, -1.36315727233887 - 4 * q
    baseline = np.where(x < .78, pre, post)
    pulse = 160 * np.maximum(0, 1 - np.abs((x - .46) / .30))
    current = polarity * scale * (baseline + q * np.round(pulse / q))
    voltage = np.where(x < .15, 20, np.where(x < .8, 14, 3))
    return pd.DataFrame({'time': np.arange(n) / 1e6, 'voltage': voltage, 'current': current})


class EdgeBackgroundTests(unittest.TestCase):
    def test_distinct_backgrounds_do_not_hide_complete_wide_event(self):
        features = extract_features(shifted_background_pulse())
        self.assertFalse(features['pulse_response_supported'])
        self.assertTrue(features['edge_pulse_response_supported'])
        self.assertGreater(features['edge_excursion_start_index'], 0)
        self.assertLess(features['edge_excursion_end_index'], features['sample_count'])
        state, reason, _ = classify_with_evidence(features)
        self.assertEqual(state, 'effective_discharge')
        self.assertEqual(reason, 'complete_excursion_outside_edge_backgrounds')

    def test_nonzero_initial_background_is_not_treated_as_open(self):
        features = extract_features(shifted_background_pulse(start_level=5))
        self.assertGreater(features['edge_pre_current_A'], 30)
        self.assertTrue(features['edge_pulse_response_supported'])
        self.assertEqual(classify_with_evidence(features)[0], 'effective_discharge')

    def test_edge_background_features_preserve_polarity_and_gain(self):
        reference = extract_features(shifted_background_pulse())
        for gain in (.01, 1, 10):
            for sign in (-1, 1):
                with self.subTest(gain=gain, sign=sign):
                    features = extract_features(shifted_background_pulse(gain, sign))
                    self.assertEqual(classify_with_evidence(features)[0], 'effective_discharge')
                    self.assertAlmostEqual(features['edge_excursion_span_fraction'],
                                           reference['edge_excursion_span_fraction'])

    def test_shape_reference_does_not_calibrate_or_change_energy(self):
        data = shifted_background_pulse()
        original = data.copy(deep=True)
        features = extract_features(data)
        pd.testing.assert_frame_equal(data, original)
        self.assertAlmostEqual(features['energy_signed_J'], trapezoid(data.voltage * data.current, data.time))
        self.assertAlmostEqual(features['energy_J'], trapezoid(abs(data.voltage * data.current), data.time))
        self.assertEqual(features['current_zero_offset_A'], 0)

    def test_single_step_stays_short_not_a_wide_pulse(self):
        features = extract_features(waveform('short'))
        self.assertFalse(features['edge_pulse_response_supported'])
        self.assertEqual(classify_with_evidence(features)[0], 'short_circuit')

    def test_monotonic_drift_is_not_complete_excursion(self):
        features = extract_features(waveform('drift'))
        self.assertFalse(features['edge_pulse_response_supported'])
        self.assertEqual(classify_with_evidence(features)[0], 'abnormal_discharge')

    def test_missing_voltage_response_cannot_support_effective_pulse(self):
        data = shifted_background_pulse()
        data['voltage'] = 20.
        features = extract_features(data)
        self.assertFalse(features['edge_pulse_response_supported'])
        self.assertNotEqual(classify_with_evidence(features)[0], 'effective_discharge')

    def test_voltage_response_after_event_is_not_borrowed(self):
        data = shifted_background_pulse()
        data['voltage'] = np.where(np.arange(len(data)) / len(data) < .82, 20., 3.)
        features = extract_features(data)
        self.assertEqual(features['edge_paired_response_fraction'], 0)
        self.assertFalse(features['edge_pulse_response_supported'])
        self.assertNotEqual(classify_with_evidence(features)[0], 'effective_discharge')

    def test_unresponsive_longest_event_does_not_hide_supported_second_event(self):
        n = 4000
        current = np.full(n, -25.)
        current[200:1800] = 71.
        current[2000:3200] = 95.
        voltage = np.full(n, 20.)
        voltage[2000:3200] = 8.
        data = pd.DataFrame({'time': np.arange(n) / 1e6, 'voltage': voltage, 'current': current})
        features = extract_features(data)
        self.assertTrue(features['edge_pulse_response_supported'])
        self.assertEqual(features['edge_excursion_start_index'], 2000)
        self.assertEqual(features['edge_excursion_end_index'], 3200)
        self.assertEqual(features['edge_supported_event_count'], 1)
        self.assertEqual(classify_with_evidence(features)[0], 'effective_discharge')

    def test_separate_short_events_do_not_pool_duration_support(self):
        n = 4000
        current = np.full(n, -25.)
        current[400:1040] = 71.
        current[1600:2240] = 95.
        voltage = np.full(n, 20.)
        voltage[400:1040] = 8.
        voltage[1600:2240] = 8.
        data = pd.DataFrame({'time': np.arange(n) / 1e6, 'voltage': voltage, 'current': current})
        features = extract_features(data)
        self.assertGreaterEqual(features['edge_candidate_event_count'], 2)
        self.assertFalse(features['edge_pulse_response_supported'])
        self.assertEqual(features['edge_supported_event_count'], 0)

    def test_unstable_edge_reference_does_not_support_new_branch(self):
        data = shifted_background_pulse()
        data.loc[:100, 'current'] = np.where(np.arange(101) % 10 < 5, -60, 60)
        features = extract_features(data)
        self.assertFalse(features['edge_reference_stable'])
        self.assertFalse(features['edge_pulse_response_supported'])

    def test_incoming_truncated_pulse_is_not_made_complete(self):
        n = 2145
        x = np.arange(n) / n
        q = 7.97869682312011
        current = -25 + q * np.round(140 * np.maximum(0, 1 - x / .7) / q)
        data = pd.DataFrame({'time': np.arange(n) / 1e6, 'voltage': np.where(x < .7, 15, 3),
                             'current': current})
        features = extract_features(data)
        self.assertFalse(features['edge_pulse_response_supported'])
        self.assertNotEqual(classify_with_evidence(features)[0], 'effective_discharge')

    def test_two_level_sparse_events_have_no_invented_noise_snr(self):
        n = 2145
        current = np.full(n, -1.36315727233887)
        for start in range(150, 2000, 65):
            current[start:start + 7] = -9.34185409545898
        data = pd.DataFrame({'time': np.arange(n) / 1e6, 'voltage': np.where(np.arange(n) < 1100, 20, 3),
                             'current': current})
        features = extract_features(data)
        self.assertFalse(features['current_scale_reliable'])
        self.assertTrue(np.isnan(features['activity_snr']))
        self.assertEqual(features['current_level_step_A'], 0)
        self.assertEqual(classify_with_evidence(features)[1], 'unreliable_current_noise_scale')

    def test_narrow_multilevel_events_are_not_made_a_broad_pulse(self):
        n = 2145
        q = 7.97869682312011
        current = np.full(n, -1.36315727233887)
        for k, start in enumerate(range(150, 2000, 65)):
            current[start:start + 7] += q * (4 if k % 2 else -4)
        data = pd.DataFrame({'time': np.arange(n) / 1e6, 'voltage': np.where(np.arange(n) < 1100, 20, 3),
                             'current': current})
        features = extract_features(data)
        self.assertFalse(features['edge_pulse_response_supported'])
        self.assertNotEqual(classify_with_evidence(features)[0], 'effective_discharge')

    def test_v3_features_require_raw_recomputation_for_v4_classification(self):
        features = extract_features(shifted_background_pulse())
        features['model_version'] = 'dynamic-screening-v3'
        with self.assertRaisesRegex(ValueError, '原始波形'):
            classify_with_evidence(features)

    def test_legacy_v3_batch_learning_is_explicitly_preserved(self):
        frame = batch_features(40)
        frame['model_version'] = 'dynamic-screening-v3'
        frame = frame.drop(columns=['edge_excursion_span_fraction', 'edge_paired_response_fraction',
                                    'edge_reference_stable', 'current_scale_reliable'])
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            result = learn_batch_states(frame, folder)
            metadata = json.loads((Path(folder) / '自学习模型.json').read_text())
            self.assertEqual(metadata['feature_model_version'], 'dynamic-screening-v3')
            self.assertEqual(set(result['model_version']), {'dynamic-screening-v3'})
            self.assertNotIn('edge_excursion_span_fraction', metadata['preprocessing']['input_features'])

    def test_mixed_feature_versions_are_rejected(self):
        frame = batch_features(40)
        frame.loc[0, 'model_version'] = 'dynamic-screening-v3'
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                learn_batch_states(frame, folder)

    def test_mismatched_source_hash_stops_before_waveform_processing(self):
        from main import main
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'data.csv'
            waveform('open').to_csv(source, index=False)
            with patch('main.load_waveform_data', side_effect=AssertionError('不应处理错误来源')), \
                 contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main([str(source), '--output', str(root / 'out'), '--expected-source-sha256', '0' * 64])
            self.assertFalse((root / 'out').exists())


if __name__ == '__main__':
    unittest.main()

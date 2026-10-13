"""针对量化背景、稀疏事件与短平台的回归，不把截图当作实验真值。"""
import unittest

import numpy as np
import pandas as pd

from modules.放电状态识别 import extract_features, classify_with_evidence


def quantized_spikes(scale=1.0, polarity=1.0):
    n = 2145
    current = np.full(n, -1.36315727233887)
    step = 7.97869682312011
    for index, start in enumerate(range(110, n - 110, 50)):
        current[start:start + 7] += step * (1 if index % 2 else -1)
    voltage = np.where(np.arange(n) < 1150, 19.5, 3.0)
    return pd.DataFrame({'time': np.arange(n) / 1e6, 'voltage': voltage,
                         'current': polarity * scale * current})


class QuantizedEventTests(unittest.TestCase):
    def test_sparse_quantized_events_are_not_sustained_oscillation(self):
        features = extract_features(quantized_spikes())
        label, reason, _ = classify_with_evidence(features)
        self.assertEqual(label, 'abnormal_discharge')
        self.assertEqual(reason, 'resolution_limited_events')
        self.assertGreater(features['noise_current'], 1)
        self.assertLess(features['oscillation_support_fraction'], .25)
        self.assertGreater(features['raw_event_count'], 10)

    def test_resolution_and_classification_preserve_polarity_and_gain(self):
        for scale in (.01, 1, 10):
            for sign in (-1, 1):
                with self.subTest(scale=scale, sign=sign):
                    features = extract_features(quantized_spikes(scale, sign))
                    self.assertAlmostEqual(features['current_level_step_A'], 7.97869682312011 * scale)
                    self.assertEqual(classify_with_evidence(features)[0], 'abnormal_discharge')

    def test_single_point_tail_is_not_evidence_of_stability(self):
        n = 2145
        index = np.arange(n)
        rng = np.random.default_rng(9)
        data = pd.DataFrame({'time': index / 1e6,
                             'voltage': np.where(index < 2014, 20, 2) + rng.normal(0, .02, n),
                             'current': np.where(index < 2014, 0, -30) + rng.normal(0, .05, n)})
        features = extract_features(data)
        self.assertLess(features['post_sample_count'], 32)
        self.assertTrue(np.isnan(features['post_current_std']))
        self.assertFalse(features['plateau_supported'])
        self.assertEqual(classify_with_evidence(features)[2], 0.0)

    def test_broad_quantized_pulse_can_have_joint_response_without_many_reversals(self):
        n = 2145
        x = np.arange(n) / n
        q = 7.97869682312011
        center = -1.36315727233887 - 3 * q
        envelope = np.clip(1 - np.abs((x - .47) / .28), 0, 1) * 160
        current = center + q * np.round(envelope / q)
        voltage = 20 - 5 * (envelope > 0) - 12 * (x > .75)
        data = pd.DataFrame({'time': np.arange(n) / 1e6, 'voltage': voltage, 'current': current})
        features = extract_features(data)
        state, reason, _ = classify_with_evidence(features)
        self.assertEqual(state, 'effective_discharge')
        self.assertEqual(reason, 'coherent_current_excursion_with_voltage_response')
        self.assertTrue(features['pulse_response_supported'])

    def test_quantized_multi_level_oscillation_remains_supported(self):
        n = 2145
        x = np.arange(n) / n
        q = 7.97869682312011
        current = -29 + q * np.round(32 * np.sin(2 * np.pi * 18 * x) / q)
        voltage = 12 + 6 * np.cos(2 * np.pi * 18 * x)
        data = pd.DataFrame({'time': np.arange(n) / 1e6, 'voltage': voltage, 'current': current})
        features = extract_features(data)
        self.assertGreater(features['current_level_step_A'], 0)
        self.assertEqual(classify_with_evidence(features)[0], 'effective_discharge')

    def test_old_feature_tables_cannot_be_silently_reclassified(self):
        features = extract_features(quantized_spikes())
        features['model_version'] = 'dynamic-screening-v2'
        with self.assertRaisesRegex(ValueError, '原始波形'):
            classify_with_evidence(features)


if __name__ == '__main__':
    unittest.main()

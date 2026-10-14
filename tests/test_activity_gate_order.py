"""全周期 RMS 不应遮蔽已有局部幅度和联合时间支持的宽事件。"""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from modules.放电状态识别 import (
    DECISION_RULE_VERSION, classify_with_evidence, extract_features,
    recognize_discharge_states,
)


def low_duty_waveform(gain=1., polarity=1., event_levels=2, response=True):
    n, q = 2145, 8.
    current = np.zeros(n)
    current[600:1201] = event_levels * q
    # 独立第三电平用于形成可观测网格；单点在分析中值滤波后被去除。
    current[50] = (1 if event_levels == 2 else 2) * q
    voltage = np.full(n, 20.)
    if response:
        voltage[600:1201] = 10.
    return pd.DataFrame({'time': np.arange(n) / 1e6, 'voltage': voltage,
                         'current': gain * polarity * current})


class ActivityGateOrderTests(unittest.TestCase):
    def test_joint_wide_event_survives_low_whole_cycle_rms(self):
        features = extract_features(low_duty_waveform())
        self.assertLess(features['activity_snr'], 4.)
        self.assertTrue(features['pulse_response_supported'])
        self.assertTrue(features['edge_pulse_response_supported'])
        self.assertEqual(classify_with_evidence(features)[0], 'effective_discharge')

    def test_low_duty_joint_event_keeps_polarity_and_gain_invariance(self):
        for gain in (.01, 1., 10.):
            for polarity in (-1., 1.):
                with self.subTest(gain=gain, polarity=polarity):
                    features = extract_features(low_duty_waveform(gain, polarity))
                    self.assertLess(features['activity_snr'], 4.)
                    self.assertEqual(classify_with_evidence(features)[0], 'effective_discharge')

    def test_current_event_without_voltage_response_stays_pending(self):
        features = extract_features(low_duty_waveform(response=False))
        self.assertEqual(classify_with_evidence(features)[1], 'resolution_limited_events')

    def test_one_grid_level_wide_event_is_not_promoted(self):
        features = extract_features(low_duty_waveform(event_levels=1))
        self.assertFalse(features['pulse_response_supported'])
        self.assertFalse(features['edge_pulse_response_supported'])
        self.assertEqual(classify_with_evidence(features)[1], 'resolution_limited_events')

    def test_separate_short_events_do_not_pool_into_wide_support(self):
        data = low_duty_waveform()
        data['current'] = 0.
        data.loc[50, 'current'] = 8.
        data.loc[400:699, 'current'] = 16.
        data.loc[1200:1500, 'current'] = 16.
        data['voltage'] = 20.
        data.loc[400:699, 'voltage'] = 10.
        data.loc[1200:1500, 'voltage'] = 10.
        features = extract_features(data)
        self.assertLess(features['activity_snr'], 4.)
        self.assertFalse(features['pulse_response_supported'])
        self.assertFalse(features['edge_pulse_response_supported'])
        self.assertNotEqual(classify_with_evidence(features)[0], 'effective_discharge')

    def test_joint_event_does_not_bypass_untrusted_scale(self):
        data = low_duty_waveform()
        data.loc[50, 'current'] = 0.
        features = extract_features(data)
        self.assertFalse(features['current_scale_reliable'])
        self.assertTrue(features['pulse_response_supported'])
        self.assertEqual(classify_with_evidence(features)[1], 'unreliable_current_noise_scale')

    def test_new_decision_revision_is_recorded_without_changing_feature_version(self):
        data = low_duty_waveform()
        cycles = pd.DataFrame({'cycle_id': [1], 'start_index': [0], 'end_index': [len(data)]})
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            result = recognize_discharge_states(data, cycles, folder)
            metadata = json.loads((Path(folder) / '识别模型参数.json').read_text())
        self.assertEqual(result.loc[0, 'model_version'], 'dynamic-screening-v4')
        self.assertEqual(result.loc[0, 'decision_rule_version'], DECISION_RULE_VERSION)
        self.assertEqual(metadata['decision_rule_version'], DECISION_RULE_VERSION)
        self.assertFalse(metadata['labels_validated'])


if __name__ == '__main__':
    unittest.main()

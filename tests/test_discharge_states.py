"""合成波形验证算法行为，不作为真实实验识别准确率。"""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/discharge-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/discharge-cache')

import numpy as np
import pandas as pd

from modules.放电状态识别 import (
    STATES, DynamicConfig, classify_state, classify_with_evidence,
    extract_features, recognize_discharge_states,
)
from modules.状态特征分析 import analyze_features
from modules.状态验证绘图 import plot_state_examples
from modules.短路候选分析 import short_candidate_analysis
from modules.短路候选绘图 import plot_short_candidates


def waveform(kind, *, n=2147, fs=1e6, seed=42, current_scale=1, sign=1):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n)
    noise_i, noise_u = rng.normal(0, .05, n), rng.normal(0, .02, n)
    if kind == 'open':
        current, voltage = noise_i, 20 + noise_u
    elif kind == 'oscillation':
        current = -29 + 4 * np.sin(2 * np.pi * 18 * x) + noise_i
        voltage = 10 + 8 * np.sin(2 * np.pi * 18 * x + .8) + noise_u
    elif kind == 'short':
        current = np.where(x < .3, 0, -30) + noise_i
        voltage = np.where(x < .3, 20, 2) + noise_u
    elif kind == 'drift':
        current, voltage = -25 - 8 * x + noise_i, 20 - 18 * x + noise_u
    else:
        raise ValueError(kind)
    return pd.DataFrame({'time': np.arange(n) / fs, 'voltage': voltage,
                         'current': sign * current_scale * current})


def state(data):
    return classify_state(extract_features(data))[0]


class DynamicStateTests(unittest.TestCase):
    def test_four_waveform_patterns(self):
        expected = dict(open='open_circuit', oscillation='effective_discharge',
                        short='short_circuit', drift='abnormal_discharge')
        for kind, label in expected.items():
            for seed in (7, 42, 91):
                with self.subTest(kind=kind, seed=seed):
                    self.assertEqual(state(waveform(kind, seed=seed)), label)

    def test_current_polarity_and_gain_invariance(self):
        for kind in ('open', 'oscillation', 'short', 'drift'):
            reference = state(waveform(kind))
            for scale in (.01, .1, 10):
                for sign in (-1, 1):
                    with self.subTest(kind=kind, scale=scale, sign=sign):
                        self.assertEqual(state(waveform(kind, current_scale=scale, sign=sign)), reference)

    def test_low_amplitude_oscillation_is_not_rejected_by_old_amp_threshold(self):
        data = waveform('oscillation', current_scale=.01)
        self.assertLess(extract_features(data)['Irms'], 3)
        self.assertEqual(state(data), 'effective_discharge')

    def test_voltage_polarity_and_gain_invariance(self):
        data = waveform('short')
        data['voltage'] *= -.1
        self.assertEqual(state(data), 'short_circuit')

    def test_negative_current_oscillation_never_becomes_short(self):
        data = waveform('oscillation')
        features = extract_features(data)
        self.assertGreater(features['current_high_ratio'], .9)
        self.assertGreater(features['reversal_count'], 4)
        self.assertEqual(classify_state(features)[0], 'effective_discharge')

    def test_constant_large_current_is_not_open(self):
        data = waveform('open')
        data['current'], data['voltage'] = -30., 2.
        self.assertEqual(state(data), 'abnormal_discharge')

    def test_measured_zero_offset_can_be_calibrated_without_changing_input(self):
        data = waveform('open')
        data['current'] -= 1.5
        original = data.copy(deep=True)
        self.assertEqual(state(data), 'abnormal_discharge')
        features = extract_features(data, DynamicConfig(current_zero_offset=-1.5))
        self.assertEqual(classify_state(features)[0], 'open_circuit')
        self.assertEqual(features['current_zero_offset_A'], -1.5)
        pd.testing.assert_frame_equal(data, original)

    def test_no_drive_is_not_claimed_as_open(self):
        data = waveform('open')
        data['current'], data['voltage'] = 0., 0.
        self.assertEqual(classify_with_evidence(extract_features(data))[1], 'no_voltage_drive_evidence')

    def test_isolated_spikes_do_not_create_discharge(self):
        data = waveform('open')
        data.loc[[100, 400, 600, 900], 'current'] = [30, -30, 25, -25]
        self.assertEqual(state(data), 'open_circuit')
        self.assertGreater(extract_features(data)['Ipeak'], 20)

    def test_jump_followed_by_oscillation_is_not_short(self):
        data = waveform('short')
        x = np.linspace(0, 1, len(data))
        data.loc[x >= .3, 'current'] += 4 * np.sin(2 * np.pi * 18 * x[x >= .3])
        self.assertEqual(state(data), 'effective_discharge')

    def test_smooth_drifting_tail_is_not_short(self):
        data = waveform('short')
        x = np.linspace(0, 1, len(data))
        data.loc[x >= .3, 'current'] -= 5 * (x[x >= .3] - .3)
        self.assertEqual(state(data), 'abnormal_discharge')

    def test_short_requires_voltage_collapse(self):
        data = waveform('short')
        data['voltage'] = 20.
        self.assertEqual(state(data), 'abnormal_discharge')

    def test_slow_oscillation_detected_by_cumulative_reversals(self):
        data = waveform('open')
        x = np.linspace(0, 1, len(data))
        data['current'] += -20 + 3 * np.sin(2 * np.pi * 3 * x)
        data['voltage'] += 3 * np.cos(2 * np.pi * 3 * x)
        self.assertEqual(state(data), 'effective_discharge')

    def test_bipolar_information_is_retained(self):
        data = waveform('oscillation')
        data['current'] += 29
        features = extract_features(data)
        self.assertGreater(features['signed_I_pp'], 1.5 * features['I_pp'])
        self.assertEqual(state(data), 'effective_discharge')

    def test_retired_features_cannot_change_classification(self):
        features = extract_features(waveform('oscillation'))
        reference = classify_state(features)
        features['current_high_ratio'], features['di_dt_max'] = 0., 1e100
        self.assertEqual(classify_state(features), reference)

    def test_sampling_rate_and_energy_use_actual_time(self):
        data = waveform('short')
        data['current'], data['voltage'] = -2., 5.
        one = extract_features(data)
        data['time'] *= 2
        two = extract_features(data)
        self.assertAlmostEqual(two['sampling_rate_Hz'], 500000, places=3)
        self.assertAlmostEqual(two['energy_J'], 2 * one['energy_J'])
        self.assertAlmostEqual(two['energy_signed_J'], -two['energy_J'])
        self.assertAlmostEqual(one['energy_J'], 10 * (len(data) - 1) / 1e6)

    def test_large_absolute_time_offset(self):
        data = waveform('short')
        data['time'] += 145.83
        self.assertEqual(state(data), 'short_circuit')

    def test_bad_data_are_reported_not_silently_repaired(self):
        cases = []
        data = waveform('short'); data.loc[100, 'current'] = np.nan
        cases.append((data, 'nonfinite_samples'))
        data = waveform('short'); data.loc[100, 'voltage'] = np.inf
        cases.append((data, 'nonfinite_samples'))
        data = waveform('short'); data.loc[100, 'time'] = data.loc[99, 'time']
        cases.append((data, 'nonmonotonic_time'))
        data = waveform('short'); data.loc[100:, 'time'] += 1e-6
        cases.append((data, 'irregular_timebase'))
        cases.append((waveform('short').iloc[:5], 'too_few_samples'))
        for data, quality in cases:
            with self.subTest(quality=quality):
                features = extract_features(data)
                self.assertEqual(features['data_quality'], quality)
                self.assertEqual(classify_state(features)[0], 'abnormal_discharge')

    def test_missing_columns_and_invalid_config(self):
        with self.assertRaises(ValueError):
            extract_features(waveform('short').drop(columns='current'))
        for kwargs in ({'median_window': 4}, {'noise_multiplier': 0}, {'analysis_windows': 1},
                       {'oscillation_persistence': 2}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                DynamicConfig(**kwargs)


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.capture = contextlib.redirect_stdout(io.StringIO())
        self.capture.__enter__()
        self.addCleanup(self.capture.__exit__, None, None, None)

    def dataset(self, repeats=1):
        data, rows, start = [], [], 0
        for kind in ('open', 'oscillation', 'short', 'drift'):
            for index in range(repeats):
                cycle = waveform(kind, seed=index + 42)
                cycle['time'] += start / 1e6
                data.append(cycle)
                rows.append({'cycle_id': len(rows) + 1, 'start_index': start, 'end_index': start + len(cycle)})
                start += len(cycle)
        return pd.concat(data, ignore_index=True), pd.DataFrame(rows)

    def test_csv_statistics_and_candidate_ranking(self):
        data, cycles = self.dataset()
        result = recognize_discharge_states(data, cycles, self.output)
        saved = pd.read_csv(self.output / '放电状态结果.csv')
        self.assertEqual(set(saved['state']), set(STATES))
        self.assertTrue(saved['review_required'].all())
        self.assertEqual(saved['start_index'].tolist(), cycles['start_index'].tolist())
        self.assertNotIn('di_dt_max', saved)
        stats = analyze_features(self.output / '放电状态结果.csv', self.output)
        self.assertIn('oscillation_index', stats['feature'].tolist())
        ranked = short_candidate_analysis(self.output / '放电状态结果.csv', self.output)
        self.assertEqual(ranked.iloc[0]['state'], 'short_circuit')
        self.assertNotIn('open_circuit', ranked['state'].tolist())
        self.assertEqual(result['classification_reason'].tolist(), saved['classification_reason'].tolist())
        parameters = json.loads((self.output / '识别模型参数.json').read_text(encoding='utf-8'))
        self.assertEqual(parameters['config']['analysis_windows'], 12)
        self.assertFalse(parameters['labels_validated'])

    def test_plots_report_class_shortages_and_preserve_review_batches(self):
        data, cycles = self.dataset()
        result = recognize_discharge_states(data, cycles, self.output)
        manifest = plot_state_examples(data, result, self.output, number=10)
        self.assertEqual(len(manifest), 4)
        for path in manifest['image_path']:
            self.assertGreater((self.output / path).stat().st_size, 1000)
        batch = (self.output / manifest.iloc[0]['image_path']).parent.parent
        summary = pd.read_csv(batch / '状态样本统计.csv')
        self.assertTrue((summary['status'] == 'insufficient').all())
        checklist_path = batch / '人工复核清单.csv'
        original = checklist_path.read_bytes()
        plot_state_examples(data, result, self.output, number=1)
        self.assertEqual(checklist_path.read_bytes(), original)
        self.assertEqual(len(list((self.output / '状态验证').glob('*/人工复核清单.csv'))), 2)

    def test_ten_examples_per_class_use_actual_cycle_ids(self):
        data, cycles = self.dataset(repeats=11)
        result = recognize_discharge_states(data, cycles, self.output)
        # 抽样逻辑独立检验；真实绘图由上一条测试执行。
        with patch('modules.状态验证绘图.plot_cycle') as draw:
            manifest = plot_state_examples(data, result, self.output, number=10)
        self.assertEqual(draw.call_count, 40)
        self.assertTrue((manifest.groupby('suggested_state').size() == 10).all())
        self.assertEqual(manifest['cycle_id'].nunique(), 40)

    def test_candidate_plotting_reuses_loaded_data_and_boundaries(self):
        data, cycles = self.dataset()
        recognize_discharge_states(data, cycles, self.output)
        short_candidate_analysis(self.output / '放电状态结果.csv', self.output)
        with patch('modules.短路候选绘图.load_waveform_data', side_effect=AssertionError('重复读取')), \
             patch('modules.短路候选绘图.detect_cycles', side_effect=AssertionError('重复检测')):
            paths = plot_short_candidates(None, self.output / '短路候选排序.csv',
                                          self.output / '候选图', 1, data=data, cycles=cycles)
        self.assertEqual(len(paths), 1)
        self.assertGreater(paths[0].stat().st_size, 1000)

    def test_invalid_boundaries_and_duplicate_ids_fail(self):
        data, cycles = self.dataset()
        for start, end in ((-1, 10), (2.5, 100), (100, 10), (0, len(data) + 1)):
            bad = cycles.copy()
            bad['start_index'] = bad['start_index'].astype(float)
            bad.loc[0, ['start_index', 'end_index']] = [start, end]
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                recognize_discharge_states(data, bad, self.output)
        cycles['cycle_id'] = 1
        with self.assertRaises(ValueError):
            recognize_discharge_states(data, cycles, self.output)

    def test_empty_cycle_results_do_not_crash_statistics(self):
        data, cycles = self.dataset()
        result = recognize_discharge_states(data, cycles.iloc[:0], self.output)
        self.assertTrue(result.empty)
        stats = analyze_features(self.output / '放电状态结果.csv', self.output)
        self.assertTrue((stats['valid_count'] == 0).all())
        self.assertTrue(short_candidate_analysis(self.output / '放电状态结果.csv', self.output).empty)

    def test_bad_cycles_keep_quality_reason_in_result(self):
        data, cycles = self.dataset()
        data.loc[cycles.iloc[2]['start_index'] + 10, 'current'] = np.nan
        result = recognize_discharge_states(data, cycles, self.output)
        bad = result.loc[result['cycle_id'] == 3].iloc[0]
        self.assertEqual(bad['data_quality'], 'nonfinite_samples')
        self.assertEqual(bad['classification_reason'], 'nonfinite_samples')
        ranked = short_candidate_analysis(self.output / '放电状态结果.csv', self.output)
        self.assertNotIn(3, ranked['cycle_id'].tolist())


if __name__ == '__main__':
    unittest.main()

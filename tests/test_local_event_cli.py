"""独立复核 CLI 的来源、边界和标签保护测试；事件拟合算法另行测试。"""
import contextlib
from dataclasses import asdict, dataclass
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from modules.放电状态识别 import (
    MODEL_VERSION, DynamicConfig, classify_with_evidence, extract_features,
)
from refine_waveform_events import _load_event_model, main, refine_waveform_events
from test_discharge_states import waveform
from test_local_events import pulse


@dataclass(frozen=True)
class MockEventConfig:
    noise_multiplier: float = 4.0


class LocalEventCliTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'source.csv'
        self.results = self.root / 'results.csv'
        self.config = self.root / '识别模型参数.json'
        self.review = self.root / 'review.csv'
        self.output = self.root / 'output'
        self.dynamic = DynamicConfig()
        blocks, rows = [], []
        for i, kind in enumerate(('drift', 'open', 'short', 'oscillation')):
            block = waveform(kind)
            start = i * len(block)
            block['time'] += 136 + start / 1e6
            blocks.append(block)
        data = pd.concat(blocks, ignore_index=True)
        data.to_csv(self.source, index=False, float_format='%.17g')
        # 采用主读取器相同的数值解析，确保来源哈希与重提比较真实有效。
        self.data = pd.read_csv(self.source)[['time', 'voltage', 'current']]
        for i, block in enumerate(blocks):
            start, end = i * len(block), (i + 1) * len(block)
            features = extract_features(self.data.iloc[start:end], self.dynamic)
            state, reason, score = classify_with_evidence(features, self.dynamic)
            rows.append({**features, 'cycle_id': i + 1, 'start_index': start,
                         'end_index': end, 'state': state, 'classification_reason': reason,
                         'short_score': score, 'automatic_state': ('undetermined' if state == 'abnormal_discharge' else state),
                         'automatic_label_validated': False,
                         'manual_state': '待定' if i == 0 else '人工保留',
                         'morphology_cluster': i, 'cluster_membership': .95})
        self.features = pd.DataFrame(rows)
        self.features.to_csv(self.results, index=False, float_format='%.17g')
        self.config.write_text(json.dumps({'model_version': MODEL_VERSION,
                                          'config': asdict(self.dynamic)}), encoding='utf-8')
        self.source_hash = hashlib.sha256(self.source.read_bytes()).hexdigest()
        samples = self.data.iloc[:len(blocks[0])].copy()
        samples['cycle_id'] = 1
        samples['source_index'] = np.arange(len(samples))
        samples['source_filename'] = self.source.name
        samples['source_sha256'] = self.source_hash
        samples['source_samples'] = len(self.data)
        samples['source_hash_verification_passed'] = True
        samples['feature_file_sha256'] = hashlib.sha256(self.results.read_bytes()).hexdigest()
        samples['calibration_applied'] = False
        self.samples = samples
        self._save_review(samples)
        self.model = ('mock-event-v1', lambda cycle, features, config=None: {
            'event_supported': True, 'event_reason': 'supported_mock',
            'event_bic_gain': 42.0,
        }, MockEventConfig())

    def _save_review(self, samples):
        samples.to_csv(self.review, index=False, float_format='%.17g')

    def _run(self, *, review=True, **kwargs):
        args = dict(results=self.results, model_config=self.config,
                    expected_source_sha256=self.source_hash, output_dir=self.output,
                    **({'review_samples': self.review} if review else {'raw_source': self.source}))
        args.update(kwargs)
        with patch('refine_waveform_events._load_event_model', return_value=self.model), \
             contextlib.redirect_stdout(io.StringIO()):
            return refine_waveform_events(**args)

    def _reject_review(self, samples, pattern):
        self._save_review(samples)
        with self.assertRaisesRegex(ValueError, pattern):
            self._run()
        self.assertFalse(self.output.exists())

    def test_full_source_reuses_boundaries_and_preserves_all_original_columns(self):
        source_bytes, result_bytes = self.source.read_bytes(), self.results.read_bytes()
        output, folder = self._run(review=False)
        self.assertEqual(output['cycle_id'].tolist(), [1, 2, 3, 4])
        self.assertEqual(output['new_dynamic_state'].tolist(),
                         ['effective_discharge', 'open_circuit', 'short_circuit', 'effective_discharge'])
        old = pd.read_csv(self.results, float_precision='round_trip')
        pd.testing.assert_frame_equal(output[old.columns].reset_index(drop=True), old, check_dtype=False)
        self.assertFalse(output['new_automatic_label_validated'].any())
        metadata = json.loads((folder / '运行来源.json').read_text())
        self.assertTrue(metadata['independent_source_sha256_verified'])
        self.assertFalse(metadata['batch_model_refitted'])
        self.assertFalse(metadata['boundaries_redetected'])
        self.assertTrue(metadata['full_batch_processed'])
        self.assertEqual(self.source.read_bytes(), source_bytes)
        self.assertEqual(self.results.read_bytes(), result_bytes)

    def test_full_source_with_real_local_model_recovers_short_complete_ramp(self):
        # 使用真实读取/拟合链路，避免 mock 只能验证列保护而漏掉算法连接错误。
        pulse(180).to_csv(self.source, index=False, float_format='%.17g')
        data = pd.read_csv(self.source)
        features = extract_features(data)
        state, reason, score = classify_with_evidence(features)
        self.assertEqual(state, 'abnormal_discharge')
        pd.DataFrame([{**features, 'cycle_id': 1, 'start_index': 0, 'end_index': len(data),
                       'state': state, 'classification_reason': reason, 'short_score': score,
                       'automatic_state': 'undetermined'}]).to_csv(self.results, index=False, float_format='%.17g')
        digest = hashlib.sha256(self.source.read_bytes()).hexdigest()
        with contextlib.redirect_stdout(io.StringIO()):
            output, folder = refine_waveform_events(
                results=self.results, model_config=self.config, expected_source_sha256=digest,
                output_dir=self.output, raw_source=self.source)
        self.assertEqual(output.loc[0, 'automatic_state'], 'undetermined')
        self.assertEqual(output.loc[0, 'new_automatic_state'], 'effective_discharge')
        self.assertTrue(output.loc[0, 'event_supported'])
        metadata = json.loads((folder / '运行来源.json').read_text())
        self.assertTrue(metadata['independent_source_sha256_verified'])
        self.assertFalse(metadata['new_automatic_labels_validated'])

    def test_review_subset_provenance_does_not_claim_independent_full_source_hash(self):
        output, folder = self._run()
        self.assertEqual(output['cycle_id'].tolist(), [1])
        self.assertEqual(output.loc[0, 'new_automatic_state'], 'effective_discharge')
        self.assertEqual(output.loc[0, 'manual_state'], '待定')
        metadata = json.loads((folder / '运行来源.json').read_text())
        self.assertFalse(metadata['independent_source_sha256_verified'])
        self.assertTrue(metadata['uploaded_manifest_matches_expected_source_sha256'])
        self.assertEqual(metadata['identity_verification'], 'selected_raw_features_and_bounds')
        self.assertFalse(metadata['full_batch_processed'])
        self.assertFalse(metadata['batch_model_refitted'])
        self.assertTrue(metadata['manifest_feature_hash_matches_provided_results'])

    def test_changed_feature_file_hash_accepted_only_after_selected_raw_feature_match(self):
        self.features['decision_rule_version'] = 'annotation-added'
        self.features.to_csv(self.results, index=False, float_format='%.17g')
        _, folder = self._run()
        metadata = json.loads((folder / '运行来源.json').read_text())
        self.assertFalse(metadata['manifest_feature_hash_matches_provided_results'])
        self.assertTrue(metadata['raw_features_match_original_results'])
        self.assertNotEqual(metadata['provided_results_sha256'], metadata['manifest_feature_file_sha256'])

    def test_dynamic_table_without_automatic_state_records_promotion(self):
        self.features.drop(columns='automatic_state').to_csv(self.results, index=False, float_format='%.17g')
        output, _ = self._run()
        self.assertNotIn('automatic_state', output)
        self.assertEqual(output.loc[0, 'new_automatic_state'], 'effective_discharge')
        self.assertTrue(output.loc[0, 'local_event_automatic_changed'])

    def test_manual_validation_record_is_preserved_and_new_label_remains_unvalidated(self):
        self.features['automatic_label_validated'] = True
        self.features['manual_comment'] = '人工确认原记录'
        self.features.to_csv(self.results, index=False, float_format='%.17g')
        output, _ = self._run()
        self.assertTrue(output.loc[0, 'automatic_label_validated'])
        self.assertFalse(output.loc[0, 'new_automatic_label_validated'])
        self.assertEqual(output.loc[0, 'manual_comment'], '人工确认原记录')

    def test_raw_feature_change_rejected_before_any_output(self):
        changed = self.samples.copy()
        changed.loc[800:1200, 'current'] += 30
        self._reject_review(changed, '特征与逐点重提不一致')

    def test_time_difference_across_sample_blocks_is_rejected(self):
        extra = self.data.iloc[2147:4294].copy()
        extra['time'] += .1
        extra['cycle_id'] = 2
        extra['source_index'] = np.arange(2147, 4294)
        for name in ('source_filename', 'source_sha256', 'source_samples',
                     'source_hash_verification_passed', 'feature_file_sha256', 'calibration_applied'):
            extra[name] = self.samples[name].iloc[0]
        self._reject_review(pd.concat([self.samples, extra], ignore_index=True),
                            '无法对应同一个时间起点')

    def test_original_features_missing_computed_descriptor_rejected(self):
        self.features.drop(columns='energy_signed_J').to_csv(self.results, index=False)
        with self.assertRaisesRegex(ValueError, '缺少原始 v4 特征: energy_signed_J'):
            self._run()
        self.assertFalse(self.output.exists())

    def test_source_manifest_mismatch_is_rejected(self):
        changed = self.samples.copy()
        changed['source_sha256'] = '0' * 64
        self._reject_review(changed, '来源 SHA-256 与预期不一致')

    def test_partial_reordered_or_duplicate_review_indices_are_rejected(self):
        for changed in (self.samples.iloc[:-1], self.samples.iloc[::-1],
                        pd.concat([self.samples, self.samples.iloc[:1]], ignore_index=True)):
            with self.subTest(rows=len(changed)):
                self._reject_review(changed, '全部样本|source_index 重复')

    def test_nonfinite_and_noninteger_sample_fields_are_rejected(self):
        for name, value in (('current', np.inf), ('time', np.nan), ('cycle_id', 1.5),
                            ('source_index', .5)):
            with self.subTest(column=name):
                changed = self.samples.copy().astype({name: float})
                changed.loc[0, name] = value
                self._reject_review(changed, '有限|整数')

    def test_boolean_and_overflow_source_indices_are_rejected(self):
        changed = self.samples.copy()
        changed['source_index'] = True
        self._reject_review(changed, '整数|布尔值')
        changed = self.samples.copy()
        changed['source_index'] = float(2 ** 63)
        self._reject_review(changed, '整数|int64')

    def test_ambiguous_mixed_or_calibrated_manifest_is_rejected(self):
        for name, value in (('source_samples', len(self.data) - 1),
                            ('source_sha256', '0' * 64),
                            ('calibration_applied', True),
                            ('source_hash_verification_passed', False)):
            with self.subTest(column=name):
                changed = self.samples.copy()
                changed.loc[0, name] = value
                self._reject_review(changed, '同一个非空值')
        for name, value in (('calibration_applied', True), ('source_hash_verification_passed', False)):
            with self.subTest(all_column=name):
                changed = self.samples.copy()
                changed[name] = value
                self._reject_review(changed, '校准|未声明')

    def test_full_source_wrong_hash_rejected_before_loader_and_output(self):
        with patch('refine_waveform_events.load_waveform_data') as loader:
            with self.assertRaisesRegex(ValueError, '原始 CSV SHA-256'):
                self._run(review=False, expected_source_sha256='0' * 64)
        loader.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_full_source_change_during_load_rejected_without_output(self):
        def changing_loader(_):
            self.source.write_bytes(self.source.read_bytes() + b'\n')
            return self.data.copy()
        with patch('refine_waveform_events.load_waveform_data', side_effect=changing_loader):
            with self.assertRaisesRegex(ValueError, '输入文件发生变化'):
                self._run(review=False)
        self.assertFalse(self.output.exists())

    def test_input_change_during_model_fit_rejected_without_output(self):
        def changing_fit(cycle, features, config=None):
            self.review.write_bytes(self.review.read_bytes() + b'\n')
            return {'event_supported': True, 'event_reason': 'supported'}
        self.model = ('mock', changing_fit, MockEventConfig())
        with self.assertRaisesRegex(ValueError, '输入文件发生变化'):
            self._run()
        self.assertFalse(self.output.exists())

    def test_incomplete_or_wrong_dynamic_configuration_is_rejected(self):
        for config in ({'model_version': MODEL_VERSION, 'config': {}},
                       {'model_version': 'dynamic-screening-v3', 'config': asdict(self.dynamic)},
                       {'model_version': MODEL_VERSION,
                        'config': {**asdict(self.dynamic), 'min_samples': 32.0}}):
            with self.subTest(config=config):
                self.config.write_text(json.dumps(config))
                with self.assertRaises(ValueError):
                    self._run()
                self.assertFalse(self.output.exists())

    def test_event_configuration_inherits_recorded_scale_and_minimum_samples(self):
        from modules.局部事件模型 import LocalEventConfig
        version, fit, config = _load_event_model(DynamicConfig(noise_multiplier=5., min_samples=40))
        self.assertTrue(version)
        self.assertTrue(callable(fit))
        self.assertEqual(config.noise_multiplier, 5.)
        self.assertEqual(config.min_segment_samples, 40)
        self.assertEqual(config.max_segments, LocalEventConfig().max_segments)
        self.assertEqual(config.max_boundary_candidates, LocalEventConfig().max_boundary_candidates)

    def test_untrusted_noise_cannot_be_promoted_by_event_model(self):
        current = self.data.iloc[:2147].copy()
        current['current'] = 0.
        current.loc[400:700, 'current'] = -30.
        features = extract_features(current, self.dynamic)
        self.assertFalse(features['current_scale_reliable'])
        state, reason, score = classify_with_evidence(features, self.dynamic)
        row = {**self.features.iloc[0].to_dict(), **features, 'state': state,
               'classification_reason': reason, 'short_score': score}
        self.features.iloc[0] = pd.Series(row)
        self.features.to_csv(self.results, index=False, float_format='%.17g')
        self.samples.loc[:, ['time', 'voltage', 'current']] = current.to_numpy()
        self._save_review(self.samples)
        output, _ = self._run()
        self.assertEqual(output.loc[0, 'new_automatic_state'], 'undetermined')
        self.assertEqual(output.loc[0, 'new_classification_reason'], 'untrusted_recomputed_current_scale')

    def test_event_model_cannot_overwrite_existing_feature_or_return_nested_output(self):
        for event in ({'event_supported': True, 'event_reason': 'x', 'state': 'short_circuit'},
                      {'event_supported': True, 'event_reason': 'x', 'event_segments': [1, 2]},
                      {'event_supported': True, 'event_reason': 'x', 'event_bic_gain': np.inf}):
            with self.subTest(event=event):
                self.model = ('mock', lambda *args, **kwargs: event, MockEventConfig())
                with self.assertRaises(ValueError):
                    self._run()
                self.assertFalse(self.output.exists())

    def test_repeated_runs_create_distinct_directories_and_preserve_prior_output(self):
        _, first = self._run()
        snapshot = (first / '局部事件复核结果.csv').read_bytes()
        _, second = self._run()
        self.assertNotEqual(first, second)
        self.assertEqual((first / '局部事件复核结果.csv').read_bytes(), snapshot)

    def test_cli_requires_exactly_one_input_and_mandatory_recorded_config(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main(['--review-samples', str(self.review), '--results', str(self.results),
                      '--expected-source-sha256', self.source_hash])
            with self.assertRaises(SystemExit):
                main([str(self.source), '--review-samples', str(self.review),
                      '--results', str(self.results), '--model-config', str(self.config),
                      '--expected-source-sha256', self.source_hash])
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()

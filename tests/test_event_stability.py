"""预算审计的来源、重演及标签保护检查；稳定性不是实验状态验证。"""
import contextlib
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from audit_event_stability import BUDGETS, audit_event_stability, main
from modules.局部事件模型 import EVENT_MODEL_VERSION, LocalEventConfig, fit_local_events
from modules.放电状态识别 import MODEL_VERSION, DynamicConfig, classify_with_evidence, extract_features
from refine_waveform_events import _load_event_model, refine_waveform_events
from test_discharge_states import waveform
from test_local_events import pulse


def mock_evidence(cycle, features, config=None):
    supported = features['Ipeak'] > 1 and config.max_segments != 10
    return {'event_model_version': EVENT_MODEL_VERSION,
            'event_supported': supported, 'event_reason': 'mock_supported' if supported else 'mock_unsupported',
            'event_boundary_truncated': False, 'event_scale_reliable': True,
            'event_structure_score_is_probability': False,
            'event_supported_count_is_proposal_count': True,
            'event_boundary_flag_is_proven_truncation': False,
            'event_config_noise_multiplier': config.noise_multiplier,
            'event_config_min_segment_samples': config.min_segment_samples,
            'event_start_index': 100 if supported else np.nan,
            'event_end_index': 500 + config.max_segments if supported else np.nan,
            'event_candidate_rejection_reasons': ''}


class EventStabilityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source, self.results = self.root / 'source.csv', self.root / 'results.csv'
        self.config_path, self.output = self.root / 'config.json', self.root / 'audit'
        self.dynamic = DynamicConfig()
        blocks = []
        for i, kind in enumerate(('drift', 'oscillation', 'open')):
            block = waveform(kind)
            block['time'] += 136 + sum(len(value) for value in blocks)/1e6
            blocks.append(block)
        pd.concat(blocks, ignore_index=True).to_csv(self.source, index=False, float_format='%.17g')
        self.data = pd.read_csv(self.source)
        self.source_hash = hashlib.sha256(self.source.read_bytes()).hexdigest()
        rows, start = [], 0
        for i, block in enumerate(blocks):
            end = start + len(block)
            features = extract_features(self.data.iloc[start:end], self.dynamic)
            state, reason, score = classify_with_evidence(features, self.dynamic)
            automatic = 'undetermined' if state == 'abnormal_discharge' else state
            new_dynamic = 'effective_discharge' if i == 0 else state
            new_automatic = 'effective_discharge' if i == 0 else automatic
            event = mock_evidence(block, features, LocalEventConfig())
            rows.append({**features, 'cycle_id': i + 1, 'start_index': start, 'end_index': end,
                         'state': state, 'classification_reason': reason, 'short_score': score,
                         'automatic_state': automatic, 'automatic_label_validated': i == 1,
                         'manual_state': '待定' if i != 1 else '人工已保留', 'manual_note': '不覆盖人工记录',
                         'morphology_cluster': i, 'cluster_membership': .9,
                         **event, 'new_dynamic_state': new_dynamic, 'new_automatic_state': new_automatic,
                         'new_automatic_label_validated': False,
                         'local_event_state_changed': state != new_dynamic,
                         'local_event_automatic_changed': automatic != new_automatic})
            start = end
        self.rows = pd.DataFrame(rows)
        self._save_rows()
        self._save_config()

    def _save_rows(self):
        self.rows.to_csv(self.results, index=False, float_format='%.17g')

    def _save_config(self):
        self.config_path.write_text(json.dumps({'model_version': MODEL_VERSION, 'config': asdict(self.dynamic)}))

    def _run(self, **overrides):
        args = dict(results=self.results, model_config=self.config_path,
                    expected_source_sha256=self.source_hash, output_dir=self.output)
        args.update(overrides)
        with patch('audit_event_stability.fit_local_events', side_effect=mock_evidence), \
             contextlib.redirect_stdout(io.StringIO()):
            return audit_event_stability(self.source, **args)

    def _reject(self, pattern, **overrides):
        with self.assertRaisesRegex(ValueError, pattern):
            self._run(**overrides)
        self.assertFalse(self.output.exists())

    def test_default_selects_new_candidates_and_preserves_every_original_column(self):
        original_bytes = [path.read_bytes() for path in (self.source, self.results, self.config_path)]
        original = pd.read_csv(self.results, float_precision='round_trip')
        summary, detail, folder = self._run()
        self.assertEqual(summary['cycle_id'].tolist(), [1])
        pd.testing.assert_frame_equal(summary[original.columns].reset_index(drop=True),
                                      original.iloc[:1].reset_index(drop=True), check_dtype=False)
        self.assertEqual(detail['trial_max_segments'].tolist(), list(BUDGETS))
        self.assertEqual(summary.loc[0, 'stability_supported_budget_count'], 3)
        self.assertFalse(summary.loc[0, 'stability_supported_all_budgets'])
        self.assertEqual(summary.loc[0, 'new_automatic_state'], 'effective_discharge')
        self.assertTrue(summary.loc[0, 'stability_baseline_replay_matches'])
        self.assertEqual(summary.loc[0, 'stability_supported_end_spread_samples'], 4)
        metadata = json.loads((folder / '运行来源.json').read_text())
        self.assertFalse(metadata['labels_changed_by_audit'])
        self.assertFalse(metadata['state_probability_computed'])
        self.assertFalse(metadata['experimental_accuracy_assessed'])
        self.assertTrue(metadata['independent_source_sha256_verified'])
        self.assertEqual([path.read_bytes() for path in (self.source, self.results, self.config_path)], original_bytes)

    def test_include_retained_automatically_selects_new_and_existing_effective(self):
        summary, detail, folder = self._run(include_retained=True)
        self.assertEqual(summary['cycle_id'].tolist(), [1, 2])
        self.assertEqual(len(detail), 8)
        self.assertEqual(summary.loc[1, 'manual_state'], '人工已保留')
        self.assertTrue(summary.loc[1, 'automatic_label_validated'])
        metadata = json.loads((folder / '运行来源.json').read_text())
        self.assertFalse(metadata['batch_model_refitted'])
        self.assertFalse(metadata['boundaries_redetected'])
        self.assertFalse(metadata['calibration_applied'])

    def test_recorded_noise_multiplier_and_min_samples_are_not_tuned(self):
        self.dynamic = DynamicConfig(noise_multiplier=5., min_samples=40)
        self._save_config()
        for index, row in self.rows.iterrows():
            block = self.data.iloc[int(row['start_index']):int(row['end_index'])]
            for name, value in extract_features(block, self.dynamic).items():
                self.rows.loc[index, name] = value
        self.rows['event_config_noise_multiplier'] = 5.
        self.rows['event_config_min_segment_samples'] = 40
        self._save_rows()
        _, detail, folder = self._run()
        self.assertTrue(detail['trial_noise_multiplier'].eq(5.).all())
        self.assertTrue(detail['trial_min_segment_samples'].eq(40).all())
        self.assertTrue(detail['trial_max_boundary_candidates'].eq(256).all())
        metadata = json.loads((folder / '运行来源.json').read_text())
        self.assertEqual(metadata['baseline_event_config']['noise_multiplier'], 5.)

    def test_empty_scope_skips_waveform_loading_but_records_real_hash_verification(self):
        self.rows.loc[0, 'new_dynamic_state'] = self.rows.loc[0, 'state']
        self.rows.loc[0, 'new_automatic_state'] = self.rows.loc[0, 'automatic_state']
        self.rows.loc[0, ['local_event_state_changed', 'local_event_automatic_changed']] = False
        self._save_rows()
        with patch('audit_event_stability.load_waveform_data') as loader:
            summary, detail, folder = self._run()
        loader.assert_not_called()
        self.assertTrue(summary.empty)
        self.assertTrue(detail.empty)
        metadata = json.loads((folder / '运行来源.json').read_text())
        self.assertTrue(metadata['empty_scope'])
        self.assertTrue(metadata['independent_source_sha256_verified'])
        self.assertIsNone(metadata['source_samples'])
        self.assertFalse(metadata['waveform_data_loaded'])
        self.assertEqual(metadata['baseline_replay_matched_cycles'], 0)

    def test_wrong_source_hash_rejected_before_loader_or_output(self):
        with patch('audit_event_stability.load_waveform_data') as loader:
            self._reject('原始 CSV SHA-256', expected_source_sha256='0'*64)
        loader.assert_not_called()

    def test_missing_old_automatic_state_or_wrong_event_version_rejected(self):
        self.rows.drop(columns='automatic_state').to_csv(self.results, index=False)
        self._reject('缺少列')
        self.rows['event_model_version'] = 'unrecorded-other-model'
        self._save_rows()
        self._reject('只支持同一')

    def test_unknown_states_or_ambiguous_booleans_rejected(self):
        for name, value in (('automatic_state', 'unknown'), ('new_automatic_state', 'unknown'),
                            ('new_dynamic_state', 'unknown'), ('event_supported', 'yes'),
                            ('local_event_state_changed', 'yes')):
            with self.subTest(name=name):
                previous = self.rows.loc[0, name]
                frame = self.rows.astype({name: object})
                frame.loc[0, name] = value
                frame.to_csv(self.results, index=False, float_format='%.17g')
                self._reject('状态|True 或 False')
                self.rows.loc[0, name] = previous

    def test_inconsistent_change_flags_rejected(self):
        self.rows.loc[0, 'local_event_state_changed'] = False
        self._save_rows()
        self._reject('与原/新动态状态不一致')

    def test_recorded_event_fields_are_all_matched_before_any_output(self):
        self.rows.loc[0, 'event_end_index'] += 1
        self._save_rows()
        self._reject('预算 6 重演.*event_end_index')

    def test_new_unrecorded_event_field_or_extra_nonempty_old_field_rejected(self):
        self.rows['event_old_unrecognized'] = 'cannot silently ignore this evidence'
        self._save_rows()
        self._reject('预算 6 重演.*event_old_unrecognized')
        self.rows = self.rows.drop(columns='event_old_unrecognized')
        self.rows = self.rows.drop(columns='event_start_index')
        self._save_rows()
        self._reject('缺少原事件证据: event_start_index')

    def test_original_v4_feature_mismatch_rejected(self):
        self.rows.loc[0, 'energy_J'] += .01
        self._save_rows()
        self._reject('原始 v4 特征与逐点重提不一致: energy_J')

    def test_sources_changing_during_trial_are_rejected_without_output(self):
        for path in (self.source, self.results, self.config_path):
            with self.subTest(path=path.name):
                before = path.read_bytes()
                def changing_fit(cycle, features, config=None):
                    result = mock_evidence(cycle, features, config)
                    if config.max_segments == 8:
                        path.write_bytes(path.read_bytes() + b'\n')
                    return result
                with patch('audit_event_stability.fit_local_events', side_effect=changing_fit), \
                     contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(ValueError, '输入文件发生变化'):
                        audit_event_stability(self.source, results=self.results, model_config=self.config_path,
                                              expected_source_sha256=self.source_hash, output_dir=self.output)
                self.assertFalse(self.output.exists())
                path.write_bytes(before)

    def test_real_local_fit_integration_preserves_promotion_and_replays_original(self):
        pulse(180).to_csv(self.source, index=False, float_format='%.17g')
        data = pd.read_csv(self.source)
        features = extract_features(data)
        state, reason, score = classify_with_evidence(features)
        self.assertEqual(state, 'abnormal_discharge')
        original_result = self.root / 'base.csv'
        pd.DataFrame([{**features, 'cycle_id': 1, 'start_index': 0, 'end_index': len(data),
                       'state': state, 'classification_reason': reason, 'short_score': score,
                       'automatic_state': 'undetermined', 'manual_note': '原人工记录保留'}]).to_csv(
                           original_result, index=False, float_format='%.17g')
        digest = hashlib.sha256(self.source.read_bytes()).hexdigest()
        with contextlib.redirect_stdout(io.StringIO()):
            local, local_folder = refine_waveform_events(
                results=original_result, model_config=self.config_path,
                expected_source_sha256=digest, output_dir=self.root / 'refinement', raw_source=self.source)
            summary, detail, folder = audit_event_stability(
                self.source, results=local_folder / '局部事件复核结果.csv', model_config=self.config_path,
                expected_source_sha256=digest, output_dir=self.output)
        self.assertEqual(summary.loc[0, 'new_automatic_state'], 'effective_discharge')
        self.assertEqual(summary.loc[0, 'automatic_state'], 'undetermined')
        self.assertEqual(summary.loc[0, 'manual_note'], '原人工记录保留')
        self.assertTrue(summary.loc[0, 'stability_baseline_replay_matches'])
        self.assertEqual(detail['trial_max_segments'].tolist(), [4, 6, 8, 10])
        self.assertTrue(detail['event_supported'].any())
        self.assertFalse(summary.loc[0, 'stability_is_experimental_validation'])

    def test_cli_requires_expected_fingerprint(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            main([str(self.source), '--results', str(self.results), '--model-config', str(self.config_path)])
        self.assertEqual(raised.exception.code, 2)


if __name__ == '__main__':
    unittest.main()

"""独立回落链对照必须先通过原 v1 来源重演，且不写回标签。"""
import contextlib
import hashlib
import io
import json
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from audit_event_stability import audit_event_stability
from modules.放电状态识别 import classify_with_evidence, extract_features
from refine_waveform_events import refine_waveform_events
import test_event_stability as fixtures
from test_local_events import pulse


MOCK_VERSION = 'mock-return-chain-shadow'


def shadow_evidence(cycle, features, config=None):
    event = fixtures.mock_evidence(cycle, features, config)
    return {**event, 'event_model_version': MOCK_VERSION, 'event_supported': True,
            'event_reason': 'shadow_supported', 'event_start_index': 100, 'event_end_index': 520}


class ReturnChainAuditTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.EventStabilityTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_comparison_preserves_original_evidence_labels_and_inherited_scales(self):
        f = self.fixture
        inputs = [p.read_bytes() for p in (f.source, f.results, f.config_path)]
        fitter = Mock(side_effect=shadow_evidence)
        with patch('audit_event_stability._load_return_chain_model', return_value=(MOCK_VERSION, fitter)):
            summary, detail, folder = f._run(compare_return_chains=True)
        original = pd.read_csv(f.results, float_precision='round_trip')
        pd.testing.assert_frame_equal(summary[original.columns].reset_index(drop=True),
                                      original.iloc[:1].reset_index(drop=True), check_dtype=False)
        self.assertEqual(summary.loc[0, 'stability_supported_budgets'], '4|6|8|10')
        self.assertTrue(summary.loc[0, 'stability_baseline_replay_matches'])
        self.assertEqual(summary.loc[0, 'stability_evidence_model_version'], MOCK_VERSION)
        self.assertTrue(summary.loc[0, 'stability_compares_return_chains'])
        self.assertTrue(detail['event_model_version'].eq(MOCK_VERSION).all())
        self.assertEqual(sorted(call.kwargs['config'].max_segments for call in fitter.call_args_list), [4, 6, 8, 10])
        self.assertTrue(all(call.kwargs['config'].noise_multiplier == 4. for call in fitter.call_args_list))
        self.assertTrue(all(call.kwargs['config'].min_segment_samples == 32 for call in fitter.call_args_list))
        self.assertEqual([p.read_bytes() for p in (f.source, f.results, f.config_path)], inputs)
        metadata = json.loads((folder/'运行来源.json').read_text())
        self.assertEqual(metadata['event_model_version'], 'local-piecewise-event-v1')
        self.assertEqual(metadata['stability_evidence_model_version'], MOCK_VERSION)
        self.assertTrue(metadata['return_chain_comparison_enabled'])
        self.assertFalse(metadata['labels_changed_by_audit'])
        self.assertFalse(metadata['audit_validates_candidate_labels'])

    def test_bad_original_replay_blocks_shadow_fit_and_all_writes(self):
        f = self.fixture
        f.rows.loc[0, 'event_reason'] = 'tampered_original_evidence'
        f._save_rows()
        fitter = Mock(side_effect=shadow_evidence)
        with patch('audit_event_stability._load_return_chain_model', return_value=(MOCK_VERSION, fitter)):
            with self.assertRaisesRegex(ValueError, '重演与原事件证据不一致'):
                f._run(compare_return_chains=True)
        fitter.assert_not_called()
        self.assertFalse(f.output.exists())

    def test_wrong_shadow_version_stops_before_writing(self):
        f = self.fixture
        with patch('audit_event_stability._load_return_chain_model',
                   return_value=(MOCK_VERSION, fixtures.mock_evidence)):
            with self.assertRaisesRegex(ValueError, '版本与本审计支持版本不一致'):
                f._run(compare_return_chains=True)
        self.assertFalse(f.output.exists())

    def test_actual_source_loader_original_model_and_chain_model_integration(self):
        f = self.fixture
        pulse(350).to_csv(f.source, index=False, float_format='%.17g')
        data = pd.read_csv(f.source)
        features = extract_features(data)
        state, reason, score = classify_with_evidence(features)
        original_result = f.root/'base.csv'
        pd.DataFrame([{**features, 'cycle_id': 1, 'start_index': 0, 'end_index': len(data),
                       'state': state, 'classification_reason': reason, 'short_score': score,
                       'automatic_state': 'undetermined', 'manual_note': '人工记录原样保留'}]).to_csv(
                           original_result, index=False, float_format='%.17g')
        digest = hashlib.sha256(f.source.read_bytes()).hexdigest()
        with contextlib.redirect_stdout(io.StringIO()):
            local, local_folder = refine_waveform_events(
                results=original_result, model_config=f.config_path, expected_source_sha256=digest,
                output_dir=f.root/'local', raw_source=f.source)
            summary, detail, folder = audit_event_stability(
                f.source, results=local_folder/'局部事件复核结果.csv', model_config=f.config_path,
                expected_source_sha256=digest, output_dir=f.output, compare_return_chains=True)
        recorded = pd.read_csv(local_folder/'局部事件复核结果.csv', float_precision='round_trip')
        pd.testing.assert_frame_equal(summary[recorded.columns], recorded, check_dtype=False)
        self.assertEqual(summary.loc[0, 'manual_note'], '人工记录原样保留')
        self.assertEqual(summary.loc[0, 'stability_evidence_model_version'], 'local-piecewise-return-chain-v2')
        self.assertTrue(detail.event_model_version.eq('local-piecewise-return-chain-v2').all())
        self.assertTrue(detail.event_supported.any())
        self.assertTrue(summary.stability_baseline_replay_matches.all())
        self.assertFalse(summary.stability_is_experimental_validation.any())

    def test_comparison_flag_requires_explicit_boolean(self):
        f = self.fixture
        with self.assertRaisesRegex(ValueError, '必须为布尔值'):
            f._run(compare_return_chains='True')
        self.assertFalse(f.output.exists())


if __name__ == '__main__':
    unittest.main()

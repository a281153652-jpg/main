import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from modules.批次自学习 import BatchConfig, learn_batch_states
from modules.放电状态识别 import MODEL_VERSION


def batch_features(n=180):
    """两个独立合成形态群，其中一个仍缺少物理命名证据。"""
    rng = np.random.default_rng(902)
    group = np.arange(n) % 2
    nudge = rng.normal(0, .015, n)
    return pd.DataFrame({
        'cycle_id': np.arange(n), 'model_version': MODEL_VERSION, 'data_quality': 'ok',
        'state': np.where(group == 0, 'effective_discharge', 'abnormal_discharge'),
        'classification_reason': np.where(group == 0, 'coherent_current_excursion_with_voltage_response',
                                           'resolution_limited_events'),
        'sample_count': 2147, 'noise_current': 1.,
        'activity_snr': np.where(group == 0, 30., 1.5) * np.exp(nudge),
        'robust_I_pp': np.where(group == 0, 120., 2.) * np.exp(nudge),
        'raw_event_fraction': np.where(group == 0, .6, .02) + nudge * .1,
        'oscillation_index': 0., 'oscillation_support_fraction': 0.,
        'bounded_excursion_span_fraction': np.where(group == 0, .5, 0.),
        'paired_response_fraction': np.where(group == 0, .45, 0.),
        'max_event_width_samples': np.where(group == 0, 1000, 3),
        'plateau_fraction': .3 + nudge,
        'post_stability_ratio': .9 + nudge,
        'transition_fraction': .2 + nudge,
        'step_bic_gain': np.where(group == 0, 500., 0.),
        'voltage_response_snr': 10. * np.exp(nudge),
        'voltage_collapse_ratio': .3 + nudge,
    })


class BatchLearningTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def run_learning(self, frame, folder='out'):
        with contextlib.redirect_stdout(io.StringIO()):
            return learn_batch_states(frame, self.root / folder)

    def test_population_selects_groups_without_forcing_four_states(self):
        result = self.run_learning(batch_features())
        metadata = json.loads((self.root / 'out' / '自学习模型.json').read_text())
        self.assertEqual(metadata['learning_status'], 'learned')
        self.assertEqual(metadata['selected_groups'], 2)
        self.assertFalse(metadata['uses_manual_labels'])
        self.assertFalse(metadata['membership_is_physical_state_probability'])
        unresolved = result['classification_reason'].eq('resolution_limited_events')
        self.assertTrue(result.loc[unresolved, 'automatic_state'].eq('undetermined').all())
        self.assertTrue(result.loc[~unresolved, 'automatic_state'].eq('effective_discharge').all())

    def test_component_number_is_not_physical_class_number(self):
        frame = batch_features()
        frame['state'] = 'effective_discharge'
        frame['classification_reason'] = 'synthetic_supported_event'
        result = self.run_learning(frame)
        self.assertEqual(result['morphology_cluster'].nunique(), 2)
        self.assertEqual(set(result['automatic_state']), {'effective_discharge'})

    def test_review_schedule_does_not_claim_physical_validation(self):
        frame = batch_features()
        frame['review_required'] = True
        result = self.run_learning(frame)
        self.assertTrue(result['dynamic_review_required'].all())
        self.assertFalse(result['automatic_label_validated'].any())
        supported = result['automatic_state'].eq('effective_discharge')
        self.assertFalse(result.loc[supported, 'review_required'].any())
        self.assertTrue(result.loc[~supported, 'review_required'].all())

    def test_row_order_does_not_change_assignments(self):
        frame = batch_features()
        a = self.run_learning(frame, 'a').set_index('cycle_id').sort_index()
        b = self.run_learning(frame.sample(frac=1, random_state=3), 'b').set_index('cycle_id').sort_index()
        np.testing.assert_array_equal(a['morphology_cluster'], b['morphology_cluster'])
        np.testing.assert_allclose(a['cluster_membership'], b['cluster_membership'])

    def test_gain_does_not_change_dimensionless_grouping(self):
        frame = batch_features()
        gain = frame.copy()
        gain['noise_current'] *= 25
        gain['robust_I_pp'] *= 25
        a = self.run_learning(frame, 'a')
        b = self.run_learning(gain, 'b')
        np.testing.assert_array_equal(a['morphology_cluster'], b['morphology_cluster'])

    def test_four_samples_do_not_claim_population_training(self):
        frame = batch_features(4)
        result = self.run_learning(frame)
        self.assertTrue(result['batch_learning_status'].eq('insufficient_cycles').all())
        self.assertTrue(result['morphology_cluster'].eq(-1).all())
        self.assertTrue(result['cluster_membership'].isna().all())

    def test_identical_batch_has_no_fabricated_groups(self):
        frame = pd.concat([batch_features(1)] * 80, ignore_index=True)
        frame['cycle_id'] = np.arange(len(frame))
        result = self.run_learning(frame)
        self.assertTrue(result['batch_learning_status'].eq('constant_features').all())
        self.assertTrue(result['cluster_membership'].isna().all())

    def test_invalid_data_is_excluded_without_zero_imputation(self):
        frame = batch_features()
        frame.loc[0, 'data_quality'] = 'nonfinite_samples'
        frame.loc[0, 'activity_snr'] = np.nan
        result = self.run_learning(frame)
        self.assertEqual(result.loc[0, 'morphology_cluster'], -1)
        self.assertEqual(result.loc[0, 'automatic_state'], 'undetermined')
        self.assertEqual(result.loc[0, 'batch_reason'], 'invalid_features_or_data')

    def test_missing_platform_is_kept_as_missing_evidence(self):
        frame = batch_features()
        frame.loc[frame.index % 2 == 1, 'post_stability_ratio'] = np.nan
        result = self.run_learning(frame)
        self.assertTrue(result.loc[result.index % 2 == 1, 'post_stability_ratio'].isna().all())
        saved = json.loads((self.root / 'out' / '自学习模型.json').read_text())
        self.assertIn('post_stability_ratio__missing', saved['preprocessing']['retained_features'])

    def test_old_feature_tables_and_duplicate_ids_are_rejected(self):
        frame = batch_features()
        frame['model_version'] = 'dynamic-screening-v2'
        with self.assertRaises(ValueError):
            self.run_learning(frame)
        frame = batch_features()
        frame.loc[1, 'cycle_id'] = frame.loc[0, 'cycle_id']
        with self.assertRaises(ValueError):
            self.run_learning(frame)

    def test_invalid_config_and_missing_feature_columns_are_rejected(self):
        with self.assertRaises(ValueError):
            BatchConfig(membership_threshold=1)
        with self.assertRaises(ValueError):
            self.run_learning(batch_features().drop(columns=['raw_event_fraction']))

    def test_input_features_and_manual_labels_are_preserved(self):
        frame = batch_features()
        frame['manual_state'] = 'user_notes'
        original = frame.copy(deep=True)
        result = self.run_learning(frame)
        pd.testing.assert_frame_equal(frame, original)
        pd.testing.assert_frame_equal(result[frame.columns], frame)
        self.assertTrue((self.root / 'out' / '自动判别统计.csv').is_file())

    def test_rarity_alone_does_not_overrule_supported_event(self):
        frame = batch_features()
        frame.loc[0, 'activity_snr'] = 1e90
        result = self.run_learning(frame)
        self.assertTrue(result.loc[0, 'distribution_outlier'])
        self.assertEqual(result.loc[0, 'automatic_state'], 'effective_discharge')
        self.assertEqual(result.loc[0, 'state'], 'effective_discharge')
        self.assertEqual(result.loc[0, 'batch_reason'], 'dynamic_evidence_with_atypical_morphology')
        self.assertTrue(result.loc[0, 'review_required'])

    def test_numerical_noise_floor_with_events_is_not_learning_evidence(self):
        frame = batch_features()
        frame['current_noise_model'] = 'measured_mad'
        frame.loc[0, 'current_noise_model'] = 'numerical_floor'
        frame.loc[0, 'noise_current'] = 1e-14
        frame.loc[0, 'activity_snr'] = 1e50
        result = self.run_learning(frame)
        self.assertEqual(result.loc[0, 'morphology_cluster'], -1)
        self.assertEqual(result.loc[0, 'automatic_state'], 'undetermined')
        self.assertEqual(result.loc[0, 'batch_reason'], 'untrusted_numerical_noise_floor')
        metadata = json.loads((self.root / 'out' / '自学习模型.json').read_text())
        self.assertEqual(metadata['excluded_numerical_noise_floor_cycles'], 1)

    def test_exact_constant_zero_is_not_rejected_as_event_scale(self):
        frame = batch_features()
        frame['current_noise_model'] = 'measured_mad'
        frame.loc[0, 'current_noise_model'] = 'numerical_floor'
        frame.loc[0, ['activity_snr', 'robust_I_pp', 'raw_event_fraction']] = 0
        frame.loc[0, 'state'] = 'open_circuit'
        result = self.run_learning(frame)
        self.assertTrue(result.loc[0, 'noise_scale_reliable_for_learning'])
        self.assertGreaterEqual(result.loc[0, 'morphology_cluster'], 0)

    def test_bounded_event_mismatch_is_audit_only(self):
        frame = batch_features()
        frame.loc[0, 'bounded_excursion_span_fraction'] = 0
        result = self.run_learning(frame)
        audit = pd.read_csv(self.root / 'out' / '判别证据审计.csv')
        self.assertTrue(audit.loc[0, 'long_raw_event_without_bounded_support'])
        self.assertEqual(result.loc[0, 'state'], frame.loc[0, 'state'])

    def test_rerun_keeps_original_dynamic_review_marker(self):
        frame = batch_features()
        frame['review_required'] = True
        a = self.run_learning(frame, 'a')
        b = self.run_learning(a, 'b')
        self.assertTrue(b['dynamic_review_required'].all())


if __name__ == '__main__':
    unittest.main()

import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from modules.周期复核导出 import export_review_cycles
from test_discharge_states import waveform
from modules.放电状态识别 import MODEL_VERSION


class ReviewExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        frames, rows = [], []
        for index, (kind, cycle_id) in enumerate(zip(('open', 'short', 'oscillation', 'drift'),
                                                    (29, 103, 570, 605))):
            frame = waveform(kind)
            start = index * len(frame)
            frame['time'] += start / 1e6
            frames.append(frame)
            rows.append({'cycle_id': cycle_id, 'start_index': start, 'end_index': start + len(frame),
                         'sample_count': len(frame), 'model_version': 'dynamic-screening-v2'})
        self.data = pd.concat(frames, ignore_index=True)
        self.results = pd.DataFrame(rows)

    def test_export_keeps_exact_values_and_source_indices(self):
        original = self.data.copy(deep=True)
        path, samples = export_review_cycles(self.data, self.results, self.root / 'out', [103, 605])
        saved = pd.read_csv(path, float_precision='round_trip')
        self.assertEqual(saved['cycle_id'].unique().tolist(), [103, 605])
        for cycle_id in (103, 605):
            row = self.results.loc[self.results.cycle_id == cycle_id].iloc[0]
            selected = saved.loc[saved.cycle_id == cycle_id]
            source = self.data.iloc[row.start_index:row.end_index]
            np.testing.assert_array_equal(selected['source_index'], np.arange(row.start_index, row.end_index))
            np.testing.assert_array_equal(selected[['time', 'voltage', 'current']], source)
        self.assertEqual(len(samples), 2 * 2147)
        pd.testing.assert_frame_equal(original, self.data)
        metadata = json.loads((path.parent / '周期复核来源.json').read_text(encoding='utf-8'))
        self.assertTrue(metadata['source_indices_end_exclusive'])

    def test_missing_id_and_invalid_boundary_are_rejected(self):
        with self.assertRaises(ValueError):
            export_review_cycles(self.data, self.results, self.root / 'missing', [999])
        bad = self.results.copy()
        bad.loc[0, 'end_index'] = len(self.data) + 1
        with self.assertRaises(ValueError):
            export_review_cycles(self.data, bad, self.root / 'bad', [29])

    def test_existing_review_files_are_preserved(self):
        path, _ = export_review_cycles(self.data, self.results, self.root / 'out', [29])
        original = path.read_bytes()
        with self.assertRaises(FileExistsError):
            export_review_cycles(self.data, self.results, self.root / 'out', [103])
        self.assertEqual(path.read_bytes(), original)

    def test_boundary_length_must_match_feature_result(self):
        bad = self.results.copy()
        bad.loc[0, 'sample_count'] -= 1
        with self.assertRaises(ValueError):
            export_review_cycles(self.data, bad, self.root / 'bad', [29])

    def test_export_cli_records_source_fingerprint(self):
        from export_review_cycles import main
        source = self.root / 'source.csv'
        result = self.root / 'result.csv'
        self.data.to_csv(source, index=False)
        self.results.to_csv(result, index=False)
        with contextlib.redirect_stdout(io.StringIO()):
            path = main([str(source), '--results', str(result), '--output', str(self.root / 'export')])
        saved = pd.read_csv(path)
        self.assertEqual(len(saved), len(self.data))
        self.assertEqual(saved.columns.tolist(),
                         ['time', 'voltage', 'current', 'cycle_id', 'source_index'])
        metadata = json.loads((path.parent / '周期复核来源.json').read_text(encoding='utf-8'))
        self.assertEqual(metadata['source_sha256'], hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertEqual(metadata['source_samples'], len(self.data))
        self.assertIsNone(metadata['source_hash_verification_passed'])

    def test_mismatched_source_hash_stops_before_loading_or_writing(self):
        from export_review_cycles import main
        source = self.root / 'source.csv'
        result = self.root / 'result.csv'
        output = self.root / 'export'
        self.data.to_csv(source, index=False)
        self.results.to_csv(result, index=False)
        with patch('export_review_cycles.load_waveform_data') as loader:
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                main([str(source), '--results', str(result), '--output', str(output),
                      '--expected-source-sha256', '0' * 64])
        loader.assert_not_called()
        self.assertFalse(output.exists())

    def test_invalid_expected_hash_stops_before_loading_or_writing(self):
        from export_review_cycles import main
        output = self.root / 'export'
        with patch('export_review_cycles.load_waveform_data') as loader, \
             contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main(['--output', str(output), '--expected-source-sha256', 'g' * 64])
        loader.assert_not_called()
        self.assertFalse(output.exists())

    def test_source_change_during_loading_stops_without_output(self):
        from export_review_cycles import main
        source = self.root / 'source.csv'
        result = self.root / 'result.csv'
        output = self.root / 'export'
        self.data.to_csv(source, index=False)
        self.results.to_csv(result, index=False)
        fingerprint = hashlib.sha256(source.read_bytes()).hexdigest()

        def changing_loader(_):
            source.write_bytes(source.read_bytes() + b'\n')
            return self.data.copy(deep=True)

        with patch('export_review_cycles.load_waveform_data', side_effect=changing_loader):
            with self.assertRaisesRegex(ValueError, '读取期间'):
                main([str(source), '--results', str(result), '--output', str(output),
                      '--expected-source-sha256', fingerprint])
        self.assertFalse(output.exists())

    def test_single_file_keeps_loaded_values_boundaries_and_source_file(self):
        from export_review_cycles import main
        source = self.root / 'source.csv'
        result = self.root / 'result.csv'
        self.data.to_csv(source, index=False, float_format='%.17g')
        self.results.to_csv(result, index=False)
        source_bytes, result_bytes = source.read_bytes(), result.read_bytes()
        fingerprint = hashlib.sha256(source_bytes).hexdigest()
        expected_loaded = pd.read_csv(source)[['time', 'voltage', 'current']]
        with contextlib.redirect_stdout(io.StringIO()):
            path = main([str(source), '--results', str(result), '--ids', '103,605',
                         '--output', str(self.root / 'export'), '--single-file',
                         '--expected-source-sha256', fingerprint.upper()])
        saved = pd.read_csv(path, float_precision='round_trip')
        core = ['time', 'voltage', 'current', 'cycle_id', 'source_index']
        self.assertEqual(saved.columns[:5].tolist(), core)
        self.assertEqual(saved['cycle_id'].unique().tolist(), [103, 605])
        for cycle_id in (103, 605):
            row = self.results.loc[self.results.cycle_id == cycle_id].iloc[0]
            selected = saved.loc[saved.cycle_id == cycle_id]
            original = expected_loaded.iloc[row.start_index:row.end_index]
            np.testing.assert_array_equal(selected[['time', 'voltage', 'current']], original)
            np.testing.assert_array_equal(selected['source_index'],
                                          np.arange(row.start_index, row.end_index))
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), fingerprint)
        self.assertEqual(result.read_bytes(), result_bytes)
        self.assertEqual(saved['source_filename'].unique().tolist(), [source.name])
        self.assertEqual(saved['source_sha256'].unique().tolist(), [fingerprint])
        self.assertEqual(saved['source_samples'].unique().tolist(), [len(self.data)])
        self.assertTrue(saved['source_hash_verification_passed'].all())
        self.assertFalse(saved['calibration_applied'].any())
        feature_hash = hashlib.sha256(result_bytes).hexdigest()
        self.assertEqual(saved['feature_file_sha256'].unique().tolist(), [feature_hash])
        metadata = json.loads((path.parent / '周期复核来源.json').read_text(encoding='utf-8'))
        self.assertEqual(metadata['expected_source_sha256'], fingerprint)
        self.assertTrue(metadata['source_hash_verification_passed'])
        self.assertEqual(metadata['feature_file_sha256'], feature_hash)

    def test_single_file_without_expected_hash_does_not_claim_verification(self):
        from export_review_cycles import main
        source = self.root / 'source.csv'
        result = self.root / 'result.csv'
        self.data.to_csv(source, index=False)
        self.results.to_csv(result, index=False)
        with contextlib.redirect_stdout(io.StringIO()):
            path = main([str(source), '--results', str(result), '--single-file',
                         '--output', str(self.root / 'export')])
        saved = pd.read_csv(path)
        self.assertTrue(saved['source_hash_verification_passed'].isna().all())
        metadata = json.loads((path.parent / '周期复核来源.json').read_text(encoding='utf-8'))
        self.assertIsNone(metadata['expected_source_sha256'])
        self.assertIsNone(metadata['source_hash_verification_passed'])

    def test_feature_snapshot_supplies_both_boundaries_and_fingerprint(self):
        from export_review_cycles import main
        source = self.root / 'source.csv'
        result = self.root / 'result.csv'
        self.data.to_csv(source, index=False)
        original_features = self.results.copy(deep=True)
        original_features['diagnostic_float'] = 1.0000000000000002
        original_features.to_csv(result, index=False, float_format='%.17g')
        snapshot = result.read_bytes()
        original_data = self.data.copy(deep=True)

        def changing_features_loader(_):
            changed = self.results.copy(deep=True)
            changed['end_index'] -= 1
            changed.to_csv(result, index=False)
            return self.data

        with patch('export_review_cycles.load_waveform_data', side_effect=changing_features_loader), \
             patch('export_review_cycles.export_review_cycles', wraps=export_review_cycles) as exporter, \
             contextlib.redirect_stdout(io.StringIO()):
            path = main([str(source), '--results', str(result), '--single-file',
                         '--output', str(self.root / 'export')])
        parsed_features = exporter.call_args.args[1]
        self.assertEqual(parsed_features.loc[0, 'diagnostic_float'], 1.0000000000000002)
        pd.testing.assert_frame_equal(parsed_features, original_features)
        pd.testing.assert_frame_equal(self.data, original_data)
        saved = pd.read_csv(path, float_precision='round_trip')
        self.assertEqual(len(saved), len(self.data))
        self.assertEqual(saved['feature_file_sha256'].unique().tolist(),
                         [hashlib.sha256(snapshot).hexdigest()])

    def test_main_reuses_auto_boundaries_and_records_source(self):
        from main import main
        source = self.root / 'source.csv'
        table = self.root / 'auto_cycles.csv'
        self.data.to_csv(source, index=False)
        self.results.to_csv(table, index=False)
        with patch('main.detect_cycles', side_effect=AssertionError('重复周期检测')), \
             patch('main.plot_state_examples'), patch('main.plot_short_candidates'), \
             contextlib.redirect_stdout(io.StringIO()):
            result = main([str(source), '--cycles', str(table), '--output', str(self.root / 'analysis'),
                           '--examples', '1'])
        self.assertEqual(result['cycle_id'].tolist(), self.results['cycle_id'].tolist())
        self.assertEqual(result['start_index'].tolist(), self.results['start_index'].tolist())
        self.assertEqual(set(result['model_version']), {MODEL_VERSION})
        metadata = json.loads((self.root / 'analysis' / '运行来源.json').read_text(encoding='utf-8'))
        self.assertEqual(metadata['source_sha256'], hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertEqual(metadata['cycles_reused_from'], str(table))


if __name__ == '__main__':
    unittest.main()

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
        metadata = json.loads((path.parent / '周期复核来源.json').read_text(encoding='utf-8'))
        self.assertEqual(metadata['source_sha256'], hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertEqual(metadata['source_samples'], len(self.data))

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
        self.assertEqual(set(result['model_version']), {'dynamic-screening-v3'})
        metadata = json.loads((self.root / 'analysis' / '运行来源.json').read_text(encoding='utf-8'))
        self.assertEqual(metadata['source_sha256'], hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertEqual(metadata['cycles_reused_from'], str(table))


if __name__ == '__main__':
    unittest.main()

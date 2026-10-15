"""新源预检的通道、全量扫描、取样边界与来源守卫。"""
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from modules.数据质量检查 import inspect_csv


class WaveformPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def write_csv(self, name, columns, rows, *, delimiter=',', bom=False):
        path = self.root / name
        with path.open('w', encoding='utf-8-sig' if bom else 'utf-8', newline='') as stream:
            writer = csv.writer(stream, delimiter=delimiter)
            writer.writerow(columns)
            writer.writerows(rows)
        return path

    def test_current_label_in_volts_is_not_selected_as_voltage_or_transformed(self):
        columns = ['extra', 'AI 5/间隙电流 (V)', '时间 (s)', 'AI 1/间隙电压 (V)']
        rows = [[9, -2, 0, 20], [9, -1, 1e-6, 10], [9, 0, 2e-6, 0]]
        path = self.write_csv('source.csv', columns, rows, bom=True)
        before = path.read_bytes()
        report = inspect_csv(path, chunk_rows=2, probe_rows=2)
        self.assertTrue(report['numeric_time_usable'])
        self.assertEqual(report['channels']['current']['column'], columns[1])
        self.assertEqual(report['channels']['voltage']['column'], columns[3])
        self.assertEqual(report['channels']['time']['column'], columns[2])
        self.assertEqual(report['channels']['current']['unit_label'], 'V')
        self.assertEqual(report['signal_statistics']['current']['minimum'], -2)
        self.assertEqual(report['signal_statistics']['current']['negative_count'], 2)
        self.assertAlmostEqual(report['sampling']['sampling_rate_from_span_Hz'], 1e6)
        self.assertFalse(report['data_checks']['physical_calibration_verified'])
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(report['source_sha256'], hashlib.sha256(before).hexdigest())
        self.assertNotIn('automatic_state', report)
        json.dumps(report, allow_nan=False)

    def test_ambiguous_semantic_channels_need_explicit_selection(self):
        path = self.write_csv('ambiguous.csv',
                              ['时间 (s)', '电压 (V)', '间隙电流 1 (A)', '间隙电流 2 (A)'],
                              [[0, 20, 1, 2], [1, 10, 2, 3]])
        with self.assertRaisesRegex(ValueError, 'current.*多个'):
            inspect_csv(path)
        report = inspect_csv(path, current_column='间隙电流 2 (A)')
        self.assertEqual(report['signal_statistics']['current']['mean'], 2.5)
        with self.assertRaisesRegex(ValueError, '不同列'):
            inspect_csv(path, time_column='时间 (s)', voltage_column='电压 (V)',
                        current_column='电压 (V)')

    def test_raw_semantic_channel_precedes_filter_and_fallback_units(self):
        path = self.write_csv('channels.csv',
                              ['时间 (s)', 'AI 1/间隙电压 (V)', '间隙电压/Filter (V)',
                               'AI 5/间隙电流 (A)', 'unused (V)'],
                              [[0, 20, 200, 1, 999], [1, 10, 100, 2, 999]])
        report = inspect_csv(path)
        self.assertEqual(report['signal_statistics']['voltage']['maximum'], 20)
        fallback = self.write_csv('fallback.csv', ['x (s)', 'u (V)', 'i (A)'],
                                  [[0, 20, 1], [1, 10, 2]], delimiter=';')
        fallback_report = inspect_csv(fallback)
        self.assertEqual(fallback_report['delimiter'], ';')
        self.assertEqual(fallback_report['channels']['voltage']['selection'], 'unit_fallback')

    def test_late_bad_values_are_scanned_even_outside_all_probes(self):
        rows = [[index / 1e6, 20, 1] for index in range(20)]
        rows[5][1] = 'broken'
        rows[12][2] = 'NaN'
        rows[14][2] = 'inf'
        rows[16][0] = ''
        path = self.write_csv('bad.csv', ['时间 (s)', '电压 (V)', '电流 (A)'], rows)
        report = inspect_csv(path, chunk_rows=3, probe_rows=2)
        self.assertFalse(report['numeric_time_usable'])
        checks = report['data_checks']['columns']
        self.assertEqual(checks['voltage']['non_numeric_count'], 1)
        self.assertEqual(checks['current']['nan_count'], 1)
        self.assertEqual(checks['current']['infinite_count'], 1)
        self.assertEqual(checks['time']['missing_count'], 1)
        self.assertEqual(report['source_samples'], 20)
        self.assertEqual(report['sampling']['interval_count'], 19)
        self.assertEqual(report['sampling']['nonfinite_interval_count'], 2)
        self.assertEqual(report['signal_statistics']['current']['finite_count'], 18)
        json.dumps(report, allow_nan=False)

    def test_time_order_checks_cross_chunk_boundaries(self):
        for name, times, expected_duplicates, expected_decreases in (
                ('duplicate', [0, 1, 1, 2, 3], 1, 0),
                ('decrease', [0, 1, .5, 2, 3], 0, 1)):
            with self.subTest(name=name):
                path = self.write_csv(name + '.csv', ['时间 (s)', '电压 (V)', '电流 (A)'],
                                      [[timestamp, 20, 1] for timestamp in times])
                report = inspect_csv(path, chunk_rows=2, probe_rows=1)
                self.assertFalse(report['numeric_time_usable'])
                self.assertEqual(report['sampling']['duplicate_time_count'], expected_duplicates)
                self.assertEqual(report['sampling']['decreasing_time_count'], expected_decreases)
                self.assertIn('non_increasing_time', report['data_checks']['blocking_reasons'])
                self.assertIsNone(report['sampling']['sampling_rate_from_span_Hz'])

    def test_empty_single_and_malformed_sources_have_distinct_outcomes(self):
        empty = self.root / 'empty.csv'
        empty.write_text('', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, '为空'):
            inspect_csv(empty)
        header_only = self.write_csv('header.csv', ['时间 (s)', '电压 (V)', '电流 (A)'], [])
        report = inspect_csv(header_only)
        self.assertFalse(report['numeric_time_usable'])
        self.assertEqual(report['source_samples'], 0)
        self.assertEqual(report['probes'], [])
        single = self.write_csv('one.csv', ['时间 (s)', '电压 (V)', '电流 (A)'], [[0, 0, 0]])
        report = inspect_csv(single)
        self.assertFalse(report['numeric_time_usable'])
        self.assertIn('fewer_than_two_samples', report['data_checks']['blocking_reasons'])
        malformed = self.write_csv('bad_width.csv', ['时间 (s)', '电压 (V)', '电流 (A)'],
                                   [[0, 20, 1], [1, 10, 2, 3]])
        with self.assertRaisesRegex(ValueError, '列数'):
            inspect_csv(malformed)
        malformed.write_text('时间 (s),电压 (V),电流 (A)\n0,20,1\n1,"10,2\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, '无法解析'):
            inspect_csv(malformed)

    def test_probe_scopes_are_real_contiguous_rows_and_never_join_their_edges(self):
        rows = [[index / 1e6, index, [0, 1, 2][index % 3]] for index in range(30)]
        path = self.write_csv('probes.csv', ['时间 (s)', '电压 (V)', '电流 (A)'], rows)
        report = inspect_csv(path, chunk_rows=4, probe_rows=6)
        self.assertEqual(report['probe_rows_total'], 18)
        self.assertEqual([(probe['start_index'], probe['end_index_exclusive'])
                          for probe in report['probes']], [(0, 6), (12, 18), (24, 30)])
        self.assertEqual(report['sampling']['probe_positive_interval_count'], 15)
        for probe in report['probes']:
            self.assertEqual(probe['sample_count'], 6)
            current = probe['signals']['current']
            self.assertEqual(current['scope'], 'this_contiguous_probe_only')
            self.assertEqual(current['distinct_level_count'], 3)
            self.assertEqual(current['observed_grid_step'], 1)
            self.assertFalse(current['hardware_adc_resolution_verified'])
        self.assertAlmostEqual(report['signal_statistics']['voltage']['mean'], 14.5)
        self.assertAlmostEqual(report['signal_statistics']['voltage']['std_population'],
                               (899 / 12) ** .5)
        self.assertEqual(report['signal_statistics']['voltage']['maximum_count'], 1)
        report_other_chunk = inspect_csv(path, chunk_rows=7, probe_rows=6)
        self.assertEqual(report['sampling'], report_other_chunk['sampling'])
        self.assertEqual(report['probes'], report_other_chunk['probes'])

    def test_source_change_during_scan_is_rejected(self):
        path = self.write_csv('changing.csv', ['时间 (s)', '电压 (V)', '电流 (A)'],
                              [[0, 20, 1], [1, 10, 2]])
        changed = False

        def progress(stage, rows):
            nonlocal changed
            if stage == 'scan_complete' and not changed:
                changed = True
                with path.open('a', encoding='utf-8') as stream:
                    stream.write('2,5,3\n')

        with self.assertRaisesRegex(ValueError, '发生变化'):
            inspect_csv(path, chunk_rows=1, probe_rows=1, progress=progress)
        self.assertTrue(changed)

    def test_unknown_time_unit_does_not_claim_hz_or_conversion(self):
        path = self.write_csv('unknown_units.csv', ['time', 'voltage', 'current'],
                              [[0, 20, -1], [1, 10, -2]])
        report = inspect_csv(path)
        self.assertTrue(report['numeric_time_usable'])
        self.assertIsNone(report['sampling']['sampling_rate_from_span_Hz'])
        self.assertEqual(report['sampling']['mean_interval_from_span'], 1)
        self.assertEqual(report['signal_statistics']['current']['mean'], -1.5)
        self.assertTrue(any('Hz' in note for note in report['notes']))

    def test_finite_large_intervals_keep_report_strict_json_serializable(self):
        path = self.write_csv('large_intervals.csv', ['时间 (s)', '电压 (V)', '电流 (A)'],
                              [[timestamp, 20, 1] for timestamp in [-1e308, 0, 1e308]])
        report = inspect_csv(path, chunk_rows=2)
        self.assertTrue(report['numeric_time_usable'])
        self.assertEqual(report['sampling']['probe_median_positive_interval'], 1e308)
        self.assertGreater(report['sampling']['sampling_rate_from_probe_median_Hz'], 0)
        json.dumps(report, allow_nan=False)


if __name__ == '__main__':
    unittest.main()

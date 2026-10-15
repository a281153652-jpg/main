"""新数据预检命令的真实文件、报告和失败行为。"""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from preflight_waveform import main


ROOT = Path(__file__).resolve().parents[1]


class PreflightCliTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root/'new_experiment.csv'
        self.output = self.root/'reports'

    def write_source(self, duplicate=False):
        rows = ['current,时间 (s),voltage']
        for i in range(64):
            t = (i-1 if duplicate and i == 48 else i)/1e6
            rows.append(f'{i % 3-1},{t:.9f},{20-i/100}')
        self.source.write_text('\n'.join(rows)+'\n', encoding='utf-8')

    def test_actual_csv_creates_unique_reports_and_keeps_input_bytes(self):
        self.write_source()
        before = self.source.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()):
            report, folder = main([str(self.source), '--output', str(self.output)])
            _, another = main([str(self.source), '--output', str(self.output)])
        self.assertNotEqual(folder, another)
        self.assertEqual(self.source.read_bytes(), before)
        recorded = json.loads((folder/'数据质量报告.json').read_text(encoding='utf-8'))
        self.assertEqual(recorded, report)
        self.assertEqual(report['source_samples'], 64)
        self.assertTrue(report['numeric_time_usable'])
        self.assertEqual(report['channels']['time']['column'], '时间 (s)')
        self.assertEqual(report['channels']['voltage']['column'], 'voltage')
        self.assertEqual(report['channels']['current']['column'], 'current')
        self.assertFalse((folder/'放电状态结果.csv').exists())

    def test_invalid_time_is_reported_with_nonzero_exit_and_no_state_file(self):
        self.write_source(duplicate=True)
        result = subprocess.run([sys.executable, '-W', 'error::FutureWarning', 'preflight_waveform.py',
                                 str(self.source), '--output', str(self.output)],
                                cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 2, result.stderr)
        paths = list(self.output.glob('*/数据质量报告.json'))
        self.assertEqual(len(paths), 1)
        self.assertFalse(json.loads(paths[0].read_text())['numeric_time_usable'])
        self.assertIn('未通过', result.stdout)
        self.assertEqual(list(self.output.glob('*/放电状态结果.csv')), [])

    def test_missing_file_explains_path_before_creating_reports(self):
        with contextlib.redirect_stderr(io.StringIO()) as errors, self.assertRaises(SystemExit) as raised:
            main([str(self.source), '--output', str(self.output)])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn('找不到输入 CSV', errors.getvalue())
        self.assertFalse(self.output.exists())

    def test_import_and_help_do_not_load_scientific_or_state_models(self):
        result = subprocess.run([sys.executable, '-c',
                                 "import sys; import preflight_waveform; "
                                 "assert 'numpy' not in sys.modules; assert 'scipy' not in sys.modules"],
                                cwd=ROOT, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, '')
        help_result = subprocess.run([sys.executable, 'preflight_waveform.py', '--help'], cwd=ROOT,
                                     capture_output=True, text=True, timeout=10)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn('--output', help_result.stdout)


if __name__ == '__main__':
    unittest.main()

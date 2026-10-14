"""检查 CLI 冷启动的等待与中断提示；不模拟放电状态。"""
import contextlib
import io
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time
import unittest

from audit_event_stability import _dependency_loading_progress


ROOT = Path(__file__).resolve().parents[1]


class AuditStartupTests(unittest.TestCase):
    def test_module_import_is_quiet_and_keeps_api_available(self):
        result = subprocess.run(
            [sys.executable, '-W', 'error::FutureWarning', '-c',
             'import audit_event_stability as a; assert callable(a.main); assert callable(a.audit_event_stability)'],
            cwd=ROOT, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, '')
        self.assertEqual(result.stderr, '')

    def test_cli_help_shows_completed_loading_before_help(self):
        result = subprocess.run(
            [sys.executable, '-W', 'error::FutureWarning', 'audit_event_stability.py', '--help'],
            cwd=ROOT, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, '')
        self.assertLess(result.stdout.index('正在加载分析依赖'), result.stdout.index('分析依赖加载完成。'))
        self.assertLess(result.stdout.index('分析依赖加载完成。'), result.stdout.index('usage:'))
        self.assertNotIn('开始校验输入', result.stdout)

    def test_sigint_during_cold_import_exits_cleanly_before_main(self):
        # 延迟真实 NumPy 导入，父进程收到标记后发送 SIGINT，无需等待安装或冷缓存。
        wrapper = '''
import builtins
import runpy
import sys
import time
original_import = builtins.__import__
def delayed_import(name, *args, **kwargs):
    if name == 'numpy':
        print('IMPORT_WAITING', flush=True)
        time.sleep(30)
    return original_import(name, *args, **kwargs)
builtins.__import__ = delayed_import
sys.argv = ['audit_event_stability.py', '--help']
runpy.run_path('audit_event_stability.py', run_name='__main__')
'''
        process = subprocess.Popen(
            [sys.executable, '-W', 'error::FutureWarning', '-c', wrapper],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        try:
            # 二进制读避免 TextIO 预读导致 select 看不到已缓冲的第二行。
            prefix = b''
            deadline = time.monotonic() + 15
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                while b'IMPORT_WAITING\n' not in prefix:
                    remaining = deadline - time.monotonic()
                    self.assertGreater(remaining, 0, '冷导入测试未收到等待标记')
                    self.assertTrue(selector.select(remaining), '冷导入测试超时')
                    chunk = process.stdout.read1(4096)
                    self.assertTrue(chunk, '冷导入测试提前退出')
                    prefix += chunk
            process.send_signal(signal.SIGINT)
            output, errors = process.communicate(timeout=10)
            stdout = (prefix + output).decode('utf-8')
            self.assertEqual(process.returncode, 130, errors.decode('utf-8'))
            self.assertIn('正在加载分析依赖', stdout)
            self.assertIn('依赖加载已中断，可重新运行', stdout)
            self.assertNotIn('分析依赖加载完成', stdout)
            self.assertNotIn('usage:', stdout)
            self.assertEqual(errors, b'')
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=5)
            process.stdout.close()
            process.stderr.close()

    def test_waiting_message_stops_when_loading_completes(self):
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            with _dependency_loading_progress(True, interval=.01):
                deadline = time.monotonic() + 2
                while '分析依赖仍在加载' not in captured.getvalue():
                    self.assertLess(time.monotonic(), deadline, '未显示等待进度')
                    time.sleep(.005)
            completed = captured.getvalue()
            time.sleep(.04)
            self.assertEqual(captured.getvalue(), completed)
        self.assertTrue(completed.rstrip().endswith('分析依赖加载完成。'))
        self.assertIn('分析依赖仍在加载', completed)


if __name__ == '__main__':
    unittest.main()

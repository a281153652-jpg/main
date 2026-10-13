"""建立独立 Python 3.12 环境；可由现有 Python 3.14 启动。"""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
TARGET = ROOT / '.venv-py312'
UV_BOOTSTRAP_VERSION = '0.12.23'


def python_in(folder):
    return folder / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def run(command, env=None):
    subprocess.run([str(value) for value in command], cwd=ROOT, env=env, check=True)


def version_of(executable):
    try:
        value = subprocess.check_output(
            [str(executable), '-c', 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'],
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return value


def is_virtual_environment(folder):
    if folder.is_symlink() or not (folder / 'pyvenv.cfg').is_file():
        return False
    try:
        payload = subprocess.check_output(
            [str(python_in(folder)), '-c',
             'import json,sys; print(json.dumps({"prefix":sys.prefix,"base":sys.base_prefix}))'],
            text=True, stderr=subprocess.DEVNULL)
        info = json.loads(payload)
        return Path(info['prefix']).resolve() == folder.resolve() and info['prefix'] != info['base']
    except (OSError, subprocess.CalledProcessError, ValueError, KeyError):
        return False


def ensure_pip(executable):
    probe = subprocess.run([str(executable), '-m', 'pip', '--version'],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if probe.returncode:
        run([executable, '-m', 'ensurepip', '--upgrade'])


def ensure_python312():
    executable = python_in(TARGET)
    # 不清理或重建已有目录，避免破坏用户环境。
    if TARGET.is_symlink():
        raise RuntimeError('.venv-py312 是符号链接，未修改；请先确认该路径用途。')
    if TARGET.exists():
        if not is_virtual_environment(TARGET) or version_of(executable) != '3.12':
            raise RuntimeError('已有 .venv-py312 不是可用的 Python 3.12 环境，未覆盖。')
        return executable
    candidates = [sys.executable, shutil.which('python3.12')]
    for candidate in candidates:
        if candidate and version_of(candidate) == '3.12':
            run([candidate, '-m', 'venv', TARGET])
            return executable
    # uv 自动获取所需解释器；工具与解释器放在仓库内独立目录，不改旧 .venv。
    runtime = ROOT / '.runtime'
    if runtime.is_symlink():
        raise RuntimeError('.runtime 是符号链接，未修改。')
    runtime.mkdir(exist_ok=True)
    env = os.environ.copy()
    env['UV_CACHE_DIR'] = str(runtime / 'uv-cache')
    env['UV_PYTHON_INSTALL_DIR'] = str(runtime / 'python')
    uv = shutil.which('uv')
    if uv is None:
        bootstrap = runtime / 'bootstrap'
        bootstrap_python = python_in(bootstrap)
        if bootstrap.is_symlink() or (bootstrap.exists() and not is_virtual_environment(bootstrap)):
            raise RuntimeError('已有工具目录不完整，未覆盖 .runtime/bootstrap。')
        if not bootstrap.exists():
            run([sys.executable, '-m', 'venv', bootstrap])
        ensure_pip(bootstrap_python)
        run([bootstrap_python, '-m', 'pip', 'install', '--no-input', f'uv=={UV_BOOTSTRAP_VERSION}'])
        uv = bootstrap / ('Scripts/uv.exe' if os.name == 'nt' else 'bin/uv')
    print('未找到系统 Python 3.12，使用 uv 获取独立解释器。', flush=True)
    run([uv, 'venv', '--python', '3.12', '--seed', TARGET], env=env)
    return executable


def main():
    if (ROOT / '.runtime').is_symlink():
        raise RuntimeError('.runtime 是符号链接，未修改。')
    executable = ensure_python312()
    if not is_virtual_environment(TARGET) or version_of(executable) != '3.12':
        raise RuntimeError('新环境版本核对失败，未安装分析依赖。')
    ensure_pip(executable)
    env = os.environ.copy()
    env['PIP_CACHE_DIR'] = str(ROOT / '.runtime' / 'pip-cache')
    env.setdefault('MPLCONFIGDIR', str(ROOT / '.runtime' / 'matplotlib'))
    env.setdefault('XDG_CACHE_HOME', str(ROOT / '.runtime' / 'cache'))
    run([executable, '-m', 'pip', 'install', '--no-input', '-r', ROOT / 'requirements.txt'], env=env)
    probe = ('import sys,pandas as pd,numpy,scipy,sklearn,matplotlib; '
             'assert sys.version_info[:2]==(3,12); '
             'assert pd.__version__=="2.2.3"; '
             'print("Python:",sys.version.split()[0]); print("pandas:",pd.__version__)')
    run([executable, '-W', 'error::FutureWarning', '-c', probe], env=env)
    print('\n独立环境已准备好：', executable)
    print('原 .venv、输入数据和输出结果均保留。')
    return executable


if __name__ == '__main__':
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f'环境准备未完成：{error}', file=sys.stderr)
        raise SystemExit(1)

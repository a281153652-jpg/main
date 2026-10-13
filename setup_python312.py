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
PIP_RECOVERY_VERSION = '26.2.1'


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


def runtime_environment():
    runtime = ROOT / '.runtime'
    if runtime.is_symlink():
        raise RuntimeError('.runtime 是符号链接，未修改。')
    runtime.mkdir(exist_ok=True)
    env = os.environ.copy()
    env['UV_CACHE_DIR'] = str(runtime / 'uv-cache')
    env['UV_PYTHON_INSTALL_DIR'] = str(runtime / 'python')
    env['PIP_CACHE_DIR'] = str(runtime / 'pip-cache')
    return env


def ensure_uv():
    env = runtime_environment()
    installed = shutil.which('uv')
    if installed:
        return installed, env
    # 用现有启动环境的 pip 安装到专用工具目录，不再依赖工具 venv 的 ensurepip。
    tools = ROOT / '.runtime' / f'uv-tools-{UV_BOOTSTRAP_VERSION}'
    marker = tools / '.setup-managed'
    if tools.is_symlink() or (tools.exists() and not tools.is_dir()):
        raise RuntimeError('独立 uv 工具路径异常，未修改。')
    if tools.exists() and any(tools.iterdir()) and not marker.is_file():
        raise RuntimeError('已有 uv 工具目录不是脚本管理的目录，未覆盖。')
    for name in ('bin', 'Scripts', 'uv'):
        if (tools / name).is_symlink():
            raise RuntimeError('uv 工具子目录是符号链接，未修改。')
    tools.mkdir(exist_ok=True)
    marker.write_text(UV_BOOTSTRAP_VERSION + '\n', encoding='utf-8')
    name = 'uv.exe' if os.name == 'nt' else 'uv'
    candidates = [tools / directory / name for directory in ('bin', 'Scripts', 'uv')]

    def tool_binary():
        for candidate in candidates:
            if candidate.is_symlink():
                raise RuntimeError('uv 可执行程序是符号链接，未修改。')
            if candidate.is_file():
                try:
                    version = subprocess.check_output([str(candidate), '--version'], text=True).split()[1]
                    if version == UV_BOOTSTRAP_VERSION:
                        return candidate
                except (OSError, subprocess.CalledProcessError, IndexError):
                    pass
        return None

    executable = tool_binary()
    if executable is None:
        run([sys.executable, '-m', 'pip', 'install', '--no-input', '--upgrade',
             '--target', tools, f'uv=={UV_BOOTSTRAP_VERSION}'], env=env)
        executable = tool_binary()
    if executable is None:
        raise RuntimeError('uv 工具安装后未找到可执行程序。')
    return executable, env


def ensure_pip(executable):
    probe = subprocess.run([str(executable), '-m', 'pip', '--version'],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not probe.returncode:
        return
    recovery = subprocess.run([str(executable), '-m', 'ensurepip', '--upgrade'],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if recovery.returncode:
        print('目标环境缺少 ensurepip，使用 uv 补齐 pip。', flush=True)
        uv, env = ensure_uv()
        run([uv, 'pip', 'install', '--python', executable, f'pip=={PIP_RECOVERY_VERSION}'], env=env)
    run([executable, '-m', 'pip', '--version'])


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
            run([candidate, '-m', 'venv', '--without-pip', TARGET])
            return executable
    # uv 自动获取所需解释器；工具与解释器放在仓库内独立目录，不改旧 .venv。
    uv, env = ensure_uv()
    print('未找到系统 Python 3.12，使用 uv 获取独立解释器。', flush=True)
    run([uv, 'venv', '--python', '3.12', TARGET], env=env)
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

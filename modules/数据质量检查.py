"""只读 CSV 技术预检；统计和局部探针均不构成放电状态或标定结论。"""
from __future__ import annotations

import csv
import hashlib
import math
import re
import time
from pathlib import Path


class _Progress:
    def __init__(self, callback):
        self.callback = callback
        self.last = -math.inf

    def emit(self, stage, rows, *, force=False):
        now = time.monotonic()
        if self.callback is not None and (force or now - self.last >= 20):
            self.callback(stage, int(rows))
            self.last = now


def _stat_signature(path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _sha256(path, progress, stage):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
            progress.emit(stage, 0)
    return digest.hexdigest()


def _header(path):
    try:
        with path.open('r', encoding='utf-8-sig', newline='') as source:
            sample = source.read(65536)
        if not sample:
            raise ValueError('CSV 文件为空')
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=',;\t').delimiter
        except csv.Error:
            # 部分最后一行不完整的取样会使 Sniffer 失败；只用首行分隔符作回退。
            first_line = sample.splitlines()[0]
            delimiter = max((',', ';', '\t'), key=first_line.count)
        with path.open('r', encoding='utf-8-sig', newline='') as source:
            names = next(csv.reader(source, delimiter=delimiter, strict=True), [])
    except (csv.Error, UnicodeError) as exc:
        raise ValueError('CSV 表头无法按 UTF-8 文本解析') from exc
    if not names or any(not name.strip() for name in names):
        raise ValueError('CSV 缺少有效列名')
    if len(set(names)) != len(names):
        raise ValueError('CSV 存在重复列名，无法唯一选择通道')
    return names, delimiter


def _semantic_roles(name):
    lowered = name.lower()
    return {
        role for role, matches in (
            ('time', ('时间' in lowered or bool(re.search(r'\btime\b|timestamp', lowered)))),
            ('voltage', ('电压' in lowered or 'voltage' in lowered)),
            ('current', ('电流' in lowered or 'current' in lowered)),
        ) if matches
    }


def _unit_label(name):
    units = re.findall(r'[（(]\s*([^()（）]+?)\s*[）)]', name)
    return units[-1].strip() if units else None


def _select_columns(names, explicit):
    selected = {}
    expected_units = {'time': ('s', 'sec', 'second', 'seconds'),
                      'voltage': ('v',), 'current': ('a',)}
    for role in ('time', 'voltage', 'current'):
        choice = explicit[role]
        selection = 'explicit'
        if choice is None:
            candidates = []
            for name in names:
                semantics = _semantic_roles(name)
                lowered = name.lower()
                filtered = any(word in lowered for word in ('filter', '滤波', '计算', 'math'))
                if role in semantics:
                    # 复合/比值通道不能凭一个关键词被当作原始通道。
                    if len(semantics) != 1:
                        continue
                    score = 30 if filtered else (50 if '间隙' in name or 'gap' in lowered else 40)
                    candidates.append((score, name, 'semantic'))
                elif not semantics and (_unit_label(name) or '').lower() in expected_units[role]:
                    candidates.append((10, name, 'unit_fallback'))
            if not candidates:
                raise ValueError(f'未识别到 {role} 通道，请显式指定列名')
            best = max(candidate[0] for candidate in candidates)
            winners = [candidate for candidate in candidates if candidate[0] == best]
            if len(winners) != 1:
                raise ValueError(f'{role} 通道有多个同等候选，请显式指定列名')
            _, choice, selection = winners[0]
        if choice not in names:
            raise ValueError(f'指定的 {role} 列不存在: {choice}')
        selected[role] = {
            'column': choice, 'unit_label': _unit_label(choice), 'selection': selection,
            'calibration_status': 'unverified',
        }
    if len({item['column'] for item in selected.values()}) != 3:
        raise ValueError('时间、电压和电流必须选择三个不同列')
    return selected


def _records(path, names, delimiter, progress, stage):
    """严格校验记录宽度；只保留投影通道，内存不随 CSV 总列数或行数增长。"""
    count = 0
    try:
        with path.open('r', encoding='utf-8-sig', newline='') as source:
            reader = csv.reader(source, delimiter=delimiter, strict=True)
            if next(reader, None) != names:
                raise ValueError('读取期间 CSV 表头发生变化')
            for record in reader:
                if not record:  # 空白行不计为数据记录；空字段记录仍参与质量检查。
                    continue
                if len(record) != len(names):
                    raise ValueError(f'CSV 第 {count + 1} 条数据记录列数与表头不一致')
                yield count, record
                count += 1
                progress.emit(stage, count)
    except (csv.Error, UnicodeError) as exc:
        raise ValueError(f'CSV 数据记录无法解析（已读取 {count} 行）') from exc


def _convert(values, np, pd):
    numeric = pd.to_numeric(pd.Series(values, dtype='object'), errors='coerce').to_numpy(dtype=float)
    counts = {'missing_count': 0, 'non_numeric_count': 0, 'nan_count': 0,
              'infinite_count': int(np.count_nonzero(np.isinf(numeric))),
              'nonfinite_count': int(np.count_nonzero(~np.isfinite(numeric)))}
    for index in np.flatnonzero(np.isnan(numeric)):
        token = values[int(index)].strip().lower()
        if not token:
            counts['missing_count'] += 1
        elif token in ('nan', '+nan', '-nan'):
            counts['nan_count'] += 1
        else:
            counts['non_numeric_count'] += 1
    return numeric, counts


class _Statistics:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.minimum = None
        self.maximum = None
        self.minimum_count = self.maximum_count = 0
        self.negative = self.zero = 0

    def update(self, values, np):
        finite = values[np.isfinite(values)]
        n = len(finite)
        if not n:
            return
        # longdouble 抑制有限极大值的聚合溢出；报告不可表示量输出 null。
        working = finite.astype(np.longdouble)
        mean = np.mean(working)
        m2 = np.sum((working - mean) ** 2)
        total = self.n + n
        delta = mean - self.mean
        self.m2 += m2 + delta * delta * self.n * n / total
        self.mean += delta * n / total
        self.n = total
        minimum, maximum = float(np.min(finite)), float(np.max(finite))
        if self.minimum is None or minimum < self.minimum:
            self.minimum, self.minimum_count = minimum, int(np.count_nonzero(finite == minimum))
        elif minimum == self.minimum:
            self.minimum_count += int(np.count_nonzero(finite == minimum))
        if self.maximum is None or maximum > self.maximum:
            self.maximum, self.maximum_count = maximum, int(np.count_nonzero(finite == maximum))
        elif maximum == self.maximum:
            self.maximum_count += int(np.count_nonzero(finite == maximum))
        self.negative += int(np.count_nonzero(finite < 0))
        self.zero += int(np.count_nonzero(finite == 0))

    def result(self, np):
        def finite_float(value):
            return float(value) if np.isfinite(value) and abs(value) <= np.finfo(float).max else None
        return {
            'scope': 'all_finite_samples', 'finite_count': self.n,
            'minimum': self.minimum, 'maximum': self.maximum,
            'mean': finite_float(self.mean) if self.n else None,
            'std_population': finite_float(np.sqrt(max(0., self.m2 / self.n))) if self.n else None,
            'negative_count': self.negative, 'zero_count': self.zero,
            'negative_fraction_of_finite': self.negative / self.n if self.n else None,
            'zero_fraction_of_finite': self.zero / self.n if self.n else None,
            'minimum_count': self.minimum_count, 'maximum_count': self.maximum_count,
            'minimum_fraction_of_finite': self.minimum_count / self.n if self.n else None,
            'maximum_fraction_of_finite': self.maximum_count / self.n if self.n else None,
        }


def _probe_specs(n, probe_rows):
    if not n:
        return []
    length = min(n, probe_rows)
    specs = []
    for label, start in (('first', 0), ('middle', (n - length) // 2), ('last', n - length)):
        duplicate = next((spec for spec in specs if spec['start_index'] == start), None)
        if duplicate is not None:
            duplicate['positions'].append(label)
        else:
            specs.append({'positions': [label], 'start_index': start,
                          'end_index_exclusive': start + length, 'sample_count': length})
    return specs


def _local_description(values, np):
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {'scope': 'this_contiguous_probe_only', 'finite_count': 0,
                'distinct_level_count': 0, 'observed_grid_step': None,
                'second_difference_mad_sigma': None, 'scale_description': 'no_finite_samples'}
    levels = np.unique(finite)
    floor = float(np.finfo(float).eps * max(1., float(np.max(np.abs(finite)))) * 32)
    step = None
    if 3 <= len(levels) <= len(finite) // 2:
        with np.errstate(over='ignore', invalid='ignore'):
            gaps = np.diff(levels)
        gaps = gaps[np.isfinite(gaps) & (gaps > floor)]
        if len(gaps) >= 2:
            candidate = float(np.min(gaps))
            with np.errstate(over='ignore', invalid='ignore'):
                multiples = gaps / candidate
            if np.all(np.isfinite(multiples)) and np.all(np.abs(multiples - np.rint(multiples)) <= 1e-6):
                step = candidate
    estimates = []
    for block in np.array_split(values, min(12, max(1, len(values) // 4))):
        if len(block) < 3:
            continue
        with np.errstate(over='ignore', invalid='ignore'):
            d2 = block[2:] - 2 * block[1:-1] + block[:-2]
        d2 = d2[np.isfinite(d2)]
        if len(d2):
            median = np.median(d2)
            estimate = float(1.4826 * np.median(np.abs(d2 - median)) / np.sqrt(6))
            if math.isfinite(estimate):
                estimates.append(estimate)
    mad = float(np.quantile(estimates, .25)) if estimates else None
    return {
        'scope': 'this_contiguous_probe_only', 'finite_count': int(len(finite)),
        'distinct_level_count': int(len(levels)), 'observed_grid_step': step,
        'second_difference_mad_sigma': mad, 'numerical_floor': floor,
        'scale_description': ('unavailable' if mad is None else
                              'numerical_floor_only' if mad <= floor else 'local_second_difference_mad'),
        'uniform_quantization_assumption_scale': step / math.sqrt(12) if step else None,
        'hardware_adc_resolution_verified': False,
    }


def inspect_csv(input_file, *, chunk_rows=200000, probe_rows=65536,
                time_column=None, voltage_column=None, current_column=None, progress=None):
    """全量检查数值/时间，首中尾有界连续探针描述电平；始终保持源文件不变。

    数值或时间问题返回 numeric_time_usable=False。通道歧义、CSV 语法/结构
    错误和扫描期间源文件变化抛 ValueError。此布尔值不代表物理标定通过。
    """
    if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, int) or chunk_rows < 1:
        raise ValueError('chunk_rows 必须为正整数')
    if isinstance(probe_rows, bool) or not isinstance(probe_rows, int) or probe_rows < 1:
        raise ValueError('probe_rows 必须为正整数')
    import numpy as np
    import pandas as pd

    path = Path(input_file).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f'找不到 CSV 文件: {path}')
    tracker = _Progress(progress)
    tracker.emit('hash_before', 0, force=True)
    initial_signature = _stat_signature(path)
    source_hash = _sha256(path, tracker, 'hash_before')
    if _stat_signature(path) != initial_signature:
        raise ValueError('源文件在初始哈希期间发生变化，请停止写入后重试')
    names, delimiter = _header(path)
    channels = _select_columns(names, {'time': time_column, 'voltage': voltage_column,
                                      'current': current_column})
    indices = [names.index(channels[role]['column']) for role in ('time', 'voltage', 'current')]
    checks = {role: {'missing_count': 0, 'non_numeric_count': 0, 'nan_count': 0,
                     'infinite_count': 0, 'nonfinite_count': 0}
              for role in ('time', 'voltage', 'current')}
    statistics = {role: _Statistics() for role in ('voltage', 'current')}
    n = 0
    previous = first = last = None
    interval_count = invalid_interval_count = duplicate_count = decreasing_count = positive_count = 0
    interval_min = interval_max = None
    positive_interval_sum = np.longdouble(0)

    def consume(rows):
        nonlocal n, previous, first, last, interval_count, invalid_interval_count
        nonlocal duplicate_count, decreasing_count, positive_count, interval_min, interval_max
        nonlocal positive_interval_sum
        converted = {}
        for role, column in zip(('time', 'voltage', 'current'), zip(*rows)):
            values, failures = _convert(list(column), np, pd)
            converted[role] = values
            for key, count in failures.items():
                checks[role][key] += count
            if role in statistics:
                statistics[role].update(values, np)
        times = converted['time']
        if n == 0:
            first = float(times[0])
        joined = times if n == 0 else np.r_[previous, times]
        with np.errstate(over='ignore', invalid='ignore'):
            differences = np.diff(joined)
        interval_count += len(differences)
        finite_dt = differences[np.isfinite(differences)]
        invalid_interval_count += int(np.count_nonzero(~np.isfinite(differences)))
        duplicate_count += int(np.count_nonzero(finite_dt == 0))
        decreasing_count += int(np.count_nonzero(finite_dt < 0))
        if len(finite_dt):
            minimum, maximum = float(np.min(finite_dt)), float(np.max(finite_dt))
            interval_min = minimum if interval_min is None else min(interval_min, minimum)
            interval_max = maximum if interval_max is None else max(interval_max, maximum)
            positive = finite_dt[finite_dt > 0]
            positive_count += len(positive)
            positive_interval_sum += np.sum(positive.astype(np.longdouble))
        n += len(times)
        previous = last = float(times[-1])

    tracker.emit('scan', 0, force=True)
    chunk = []
    for _, record in _records(path, names, delimiter, tracker, 'scan'):
        chunk.append(tuple(record[index] for index in indices))
        if len(chunk) == chunk_rows:
            consume(chunk)
            chunk = []
    if chunk:
        consume(chunk)
    tracker.emit('scan_complete', n, force=True)

    specs = _probe_specs(n, probe_rows)
    probe_values = [[[], [], []] for _ in specs]
    tracker.emit('probes', 0, force=True)
    for row_index, record in _records(path, names, delimiter, tracker, 'probes'):
        for spec, values in zip(specs, probe_values):
            if spec['start_index'] <= row_index < spec['end_index_exclusive']:
                for column, source_index in zip(values, indices):
                    column.append(record[source_index])
    probes, sampled_intervals = [], []
    for spec, values in zip(specs, probe_values):
        if len(values[0]) != spec['sample_count']:
            raise ValueError('源文件行数在连续探针扫描期间发生变化')
        arrays = [_convert(column, np, pd)[0] for column in values]
        with np.errstate(over='ignore', invalid='ignore'):
            differences = np.diff(arrays[0])
        sampled_intervals.extend(differences[np.isfinite(differences) & (differences > 0)].tolist())
        probes.append({**spec, 'scope': 'contiguous_source_rows_zero_based',
                       'signals': {role: _local_description(array, np)
                                   for role, array in zip(('voltage', 'current'), arrays[1:])}})

    tracker.emit('hash_after', n, force=True)
    final_hash = _sha256(path, tracker, 'hash_after')
    if final_hash != source_hash or _stat_signature(path) != initial_signature:
        raise ValueError('源文件在预检期间发生变化，未产生可信预检结果；请停止写入后重试')
    reasons = []
    if n < 2:
        reasons.append('fewer_than_two_samples')
    for role in checks:
        if checks[role]['nonfinite_count']:
            reasons.append(f'nonfinite_or_non_numeric_{role}')
    if duplicate_count or decreasing_count:
        reasons.append('non_increasing_time')
    if invalid_interval_count:
        reasons.append('nonfinite_time_interval')
    usable = not reasons
    mean_positive = float(positive_interval_sum / positive_count) if positive_count else None
    # 偶数个极大但有限的间隔取中位数时，float64 中间求和可能溢出。
    probe_median = float(np.median(np.asarray(sampled_intervals, dtype=np.longdouble))) if sampled_intervals else None
    time_unit = (channels['time']['unit_label'] or '').lower()
    seconds_declared = time_unit in ('s', 'sec', 'second', 'seconds')
    mean_interval = float((np.longdouble(last) - np.longdouble(first)) / (n - 1)) if usable else None

    def rate_in_hz(interval):
        if not seconds_declared or interval is None or interval <= 0:
            return None
        rate = np.longdouble(1) / np.longdouble(interval)
        return float(rate) if rate <= np.finfo(float).max else None

    notes = [
        '本预检只检查技术可读性、数值和时间；不确认传感器标定、零点、电流方向或加工状态。',
        '未对原信号取绝对值、归零、缩放、滤波、重采样或改写；负值本身不是坏数据证据。',
        '极值占用仅为全量描述，不据此认定削顶、短路或硬件饱和。',
        '局部电平网格和差分尺度仅描述所列连续窗口，不能推断硬件 ADC 分辨率或全量噪声。',
        '未复用其他源文件的周期、群组、分类或人工确认标签。',
    ]
    if not seconds_declared:
        notes.append('时间列未明确声明秒单位；不自动换算，Hz 采样率留空，间隔值保留原时间单位。')
    for role, expected in (('voltage', 'v'), ('current', 'a')):
        if (channels[role]['unit_label'] or '').lower() != expected:
            notes.append(f'{role} 列单位未明确声明为 {expected.upper()}，缩放需核对，未自动转换。')
    result = {
        'report_version': 'waveform-preflight-v1',
        'source_filename': path.name, 'source_path': str(path),
        'source_sha256': source_hash, 'source_sha256_after': final_hash,
        'source_unchanged': True, 'source_size_bytes': initial_signature[2],
        'source_samples': n, 'column_names': names, 'delimiter': delimiter,
        'numeric_time_usable': usable, 'channels': channels,
        'sampling': {
            'scope': 'all_source_rows', 'interval_count': interval_count,
            'nonfinite_interval_count': invalid_interval_count,
            'duplicate_time_count': duplicate_count, 'decreasing_time_count': decreasing_count,
            'positive_interval_count': positive_count,
            'first_time_value': first if first is not None and math.isfinite(first) else None,
            'last_time_value': last if last is not None and math.isfinite(last) else None,
            'declared_time_unit': channels['time']['unit_label'],
            'minimum_finite_interval': interval_min, 'maximum_finite_interval': interval_max,
            'mean_positive_interval': mean_positive,
            'mean_interval_from_span': mean_interval,
            'sampling_rate_from_span_Hz': rate_in_hz(mean_interval),
            'probe_median_positive_interval': probe_median,
            'sampling_rate_from_probe_median_Hz': rate_in_hz(probe_median),
            'probe_interval_scope': 'within_each_listed_probe_only; no_join_between_probes',
            'probe_positive_interval_count': len(sampled_intervals),
        },
        'data_checks': {'scope': 'all_source_rows', 'columns': checks,
                        'blocking_reasons': reasons, 'physical_calibration_verified': False},
        'signal_statistics': {role: stats.result(np) for role, stats in statistics.items()},
        'probes': probes, 'probe_rows_limit_per_window': probe_rows,
        'probe_rows_total': sum(spec['sample_count'] for spec in specs),
        'notes': notes,
    }
    tracker.emit('complete', n, force=True)
    return result

"""噪声自适应的物理状态筛查，输出均需人工复核，非已训练的机器学习模型。"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.ndimage import median_filter

STATES = ('open_circuit', 'effective_discharge', 'short_circuit', 'abnormal_discharge')
MODEL_VERSION = 'dynamic-screening-v2'


@dataclass(frozen=True)
class DynamicConfig:
    """统计/形态参数，不使用固定安培或伏特阈值。"""
    noise_multiplier: float = 4.0
    median_window: int = 5
    analysis_windows: int = 12
    min_samples: int = 32
    min_reversals: int = 4
    oscillation_persistence: float = 0.25
    min_plateau_fraction: float = 0.4
    voltage_collapse_ratio: float = 0.5
    current_zero_offset: float = 0.0

    def __post_init__(self):
        if not np.isfinite(self.current_zero_offset):
            raise ValueError('current_zero_offset 必须为有限数值')
        if self.noise_multiplier <= 0 or not np.isfinite(self.noise_multiplier):
            raise ValueError('noise_multiplier 必须为有限正数')
        if self.median_window < 1 or self.median_window % 2 != 1:
            raise ValueError('median_window 必须为正奇数')
        if self.min_samples < 8 or self.analysis_windows < 2 or self.min_reversals < 2:
            raise ValueError('样本数、窗口数或反转数配置不合法')
        for value in (self.oscillation_persistence, self.min_plateau_fraction,
                      self.voltage_collapse_ratio):
            if not 0 < value <= 1:
                raise ValueError('比例参数必须位于 (0, 1]')


def _noise_sigma(values, windows):
    """局部二阶差分 MAD / sqrt(6)，下四分位降低边沿/动态区污染。"""
    estimates = []
    for block in np.array_split(values, min(windows, len(values) // 4)):
        diff = np.diff(block, n=2)
        estimates.append(1.4826 * np.median(np.abs(diff - np.median(diff))) / np.sqrt(6))
    floor = np.finfo(float).eps * max(1.0, float(np.max(np.abs(values)))) * 32
    return max(float(np.quantile(estimates, 0.25)), floor)


def _reversal_indices(values, excursion):
    """累计反向变化超过噪声界才记一次转折，不依赖相邻点导数峰值。"""
    extreme, extreme_index, direction = values[0], 0, 0
    reversals = []
    for index in range(1, len(values)):
        value = values[index]
        if direction == 0:
            if abs(value - extreme) > excursion:
                direction = 1 if value > extreme else -1
                extreme, extreme_index = value, index
        elif direction * (value - extreme) >= 0:
            extreme, extreme_index = value, index
        elif direction * (extreme - value) > excursion:
            reversals.append(extreme_index)
            direction = -direction
            extreme, extreme_index = value, index
    return np.asarray(reversals, dtype=int)


def _step_metrics(current, voltage, sigma_i, sigma_u, config):
    """单变点常值模型与线性/常值模型比较，再检查跃迁后的联合稳定性。"""
    n = len(current)
    guard = max(2, n // 20)
    candidates = np.arange(guard, n - guard)
    centered = current - np.mean(current)  # 居中避免大直流偏置下的平方和消减。
    sums, squares = np.cumsum(centered), np.cumsum(centered ** 2)
    left = squares[candidates - 1] - sums[candidates - 1] ** 2 / candidates
    right = (squares[-1] - squares[candidates - 1]
             - (sums[-1] - sums[candidates - 1]) ** 2 / (n - candidates))
    errors = np.maximum(left + right, 0)
    best = int(np.argmin(errors))
    split = int(candidates[best])
    x = np.linspace(-1, 1, n)
    slope = float(np.dot(centered, x) / np.dot(x, x))
    alternative_sse = min(float(squares[-1]), float(np.sum((centered - slope * x) ** 2)))
    regularizer = n * sigma_i ** 2
    bic_gain = n * np.log((alternative_sse + regularizer) / (errors[best] + regularizer)) - np.log(n)
    # 排除端部复位和过渡带；平台实际长度另行输出，不把末端小段当长平台。
    before = current[guard:max(guard + 1, split - guard)]
    after = current[min(split + guard, n - guard - 1):n - guard]
    u_before = np.abs(voltage[guard:max(guard + 1, split - guard)])
    u_after = np.abs(voltage[min(split + guard, n - guard - 1):n - guard])
    pre, post = float(np.median(before)), float(np.median(after))
    u_pre, u_post = float(np.median(u_before)), float(np.median(u_after))
    delta = post - pre
    if abs(delta) > config.noise_multiplier * sigma_i:
        progress = (current - pre) / delta
        low = np.flatnonzero(progress[:split] <= 0.1)
        high = np.flatnonzero(progress[split:] >= 0.9)
        width = ((split + high[0] if len(high) else n - 1)
                 - (low[-1] if len(low) else 0)) / n
    else:
        width = 1.0
    return {
        'step_index': split, 'step_bic_gain': float(bic_gain),
        'I_jump': abs(post) - abs(pre), 'jump_snr': (abs(post) - abs(pre)) / sigma_i,
        'transition_fraction': float(width), 'plateau_fraction': len(after) / n,
        'post_current_std': float(np.std(after)), 'post_current_snr': abs(post) / sigma_i,
        'post_stability_ratio': float(np.mean(np.abs(after - post) <= config.noise_multiplier * sigma_i)),
        'voltage_drop_snr': (u_pre - u_post) / sigma_u,
        'voltage_collapse_ratio': u_post / max(u_pre, sigma_u),
        'post_voltage_std': float(np.std(u_after)),
    }


def extract_features(cycle, config=None):
    """输入 time(s)/voltage(V)/current(A)，不插值/删除坏点，不硬编码采样率。"""
    config = config or DynamicConfig()
    for column in ('time', 'voltage', 'current'):
        if column not in cycle:
            raise ValueError(f'缺少必需列: {column}')
    features = {'model_version': MODEL_VERSION, 'sample_count': len(cycle), 'data_quality': 'ok',
                'current_zero_offset_A': config.current_zero_offset}
    if len(cycle) < config.min_samples:
        features['data_quality'] = 'too_few_samples'
        return features
    time = cycle['time'].to_numpy(dtype=float)
    voltage = cycle['voltage'].to_numpy(dtype=float)
    current = cycle['current'].to_numpy(dtype=float) - config.current_zero_offset
    if not all(np.all(np.isfinite(v)) for v in (time, voltage, current)):
        features['data_quality'] = 'nonfinite_samples'
        return features
    dt = np.diff(time)
    if np.any(dt <= 0):
        features['data_quality'] = 'nonmonotonic_time'
        return features
    sample_interval = float(np.median(dt))
    if np.max(np.abs(dt / sample_interval - 1)) > 0.01:
        features['data_quality'] = 'irregular_timebase'
        return features
    sigma_i = _noise_sigma(current, config.analysis_windows)
    sigma_u = _noise_sigma(voltage, config.analysis_windows)
    filtered = median_filter(current, size=config.median_window, mode='nearest')
    filtered_u = median_filter(voltage, size=config.median_window, mode='nearest')
    magnitude, filtered_magnitude = np.abs(current), np.abs(filtered)
    excursion = config.noise_multiplier * np.sqrt(2) * sigma_i
    guard = max(1, len(cycle) // 20)
    interior = filtered[guard:-guard]
    reversals = _reversal_indices(interior, excursion)
    occupied = np.unique(reversals * config.analysis_windows // len(interior))
    span = (reversals[-1] - reversals[0]) / len(interior) if len(reversals) > 1 else 0.0
    low_voltage_limit = config.voltage_collapse_ratio * np.quantile(np.abs(voltage), 0.9)
    high = magnitude > 0.5 * np.max(magnitude)
    features.update({
        'sampling_rate_Hz': 1 / sample_interval, 'duration_s': float(time[-1] - time[0]),
        'noise_current': sigma_i, 'noise_voltage': sigma_u,
        'Ipeak': float(np.max(magnitude)), 'Irms': float(np.sqrt(np.mean(current ** 2))),
        'I_pp': float(np.ptp(magnitude)), 'I_std': float(np.std(magnitude)),
        'signed_I_pp': float(np.ptp(current)), 'signed_I_std': float(np.std(current)),
        'robust_I_pp': float(np.quantile(filtered, 0.99) - np.quantile(filtered, 0.01)),
        'activity_snr': float(np.sqrt(np.mean(filtered_magnitude ** 2)) / sigma_i),
        'voltage_snr': float(np.sqrt(np.mean(filtered_u ** 2)) / sigma_u),
        'voltage_response_snr': float(np.std(filtered_u) / sigma_u),
        'change_count': int(np.count_nonzero(np.abs(np.diff(interior)) > excursion)),
        'reversal_count': len(reversals), 'oscillation_index': len(occupied) / config.analysis_windows,
        'oscillation_span_ratio': float(span),
        'Umin': float(np.min(voltage)), 'voltage_std': float(np.std(voltage)),
        'voltage_low_ratio': float(np.mean(np.abs(voltage) < low_voltage_limit)),
        'discharge_ratio': float(np.mean(filtered_magnitude > config.noise_multiplier * sigma_i)),
        # 保留绝对功率积分作为活动能量代理；不等于校准后的净加工能量。
        'energy_J': float(trapezoid(np.abs(voltage * current), time)),
        'energy_signed_J': float(trapezoid(voltage * current, time)),
        # 兼容旧分析表的描述量，分类与候选排序都不使用。
        'current_high_ratio': float(np.mean(high)),
        'current_high_std': float(np.std(magnitude[high])) if np.any(high) else 0.0,
        'current_high_time': float(np.count_nonzero(high) * sample_interval),
    })
    features.update(_step_metrics(filtered, filtered_u, sigma_i, sigma_u, config))
    return features


def classify_with_evidence(features, config=None):
    """返回四类建议、原因、短路条件满足比例；分数不是校准概率。"""
    config = config or DynamicConfig()
    if features['data_quality'] != 'ok':
        return 'abnormal_discharge', features['data_quality'], 0.0
    z = config.noise_multiplier
    oscillatory = (features['reversal_count'] >= config.min_reversals
                   and features['oscillation_index'] >= config.oscillation_persistence
                   and features['oscillation_span_ratio'] >= config.oscillation_persistence)
    if features['activity_snr'] <= z:
        if features['voltage_snr'] > z:
            return 'open_circuit', 'current_at_noise_floor', 0.0
        return 'abnormal_discharge', 'no_voltage_drive_evidence', 0.0
    evidence = (
        features['step_bic_gain'] > 0,
        features['jump_snr'] > z,
        features['transition_fraction'] <= 1 / config.analysis_windows,
        features['plateau_fraction'] >= config.min_plateau_fraction,
        features['post_current_snr'] > z,
        features['post_current_std'] <= z * features['noise_current'],
        features['voltage_drop_snr'] > z,
        features['voltage_collapse_ratio'] <= config.voltage_collapse_ratio,
        features['post_voltage_std'] <= z * features['noise_voltage'],
        not oscillatory,
    )
    short_score = sum(bool(v) for v in evidence) / len(evidence)
    if all(evidence):
        return 'short_circuit', 'joint_step_then_stable_plateau', short_score
    if oscillatory and features['voltage_response_snr'] > z:
        return 'effective_discharge', 'sustained_oscillation_with_voltage_response', short_score
    reason = 'oscillation_without_voltage_response' if oscillatory else 'active_without_clear_dynamic_pattern'
    return 'abnormal_discharge', reason, short_score


def classify_state(features, config=None):
    """保留原二元返回接口；异常是独立主状态。"""
    state, _, _ = classify_with_evidence(features, config)
    sub_state = {'effective_discharge': 'normal', 'abnormal_discharge': 'abnormal'}.get(state, 'none')
    return state, sub_state


def recognize_discharge_states(data, cycles, output_dir, config=None):
    config = config or DynamicConfig()
    required = {'cycle_id', 'start_index', 'end_index'}
    if not required.issubset(cycles.columns):
        raise ValueError(f'周期表缺少列: {sorted(required - set(cycles.columns))}')
    if cycles['cycle_id'].duplicated().any():
        raise ValueError('cycle_id 不得重复')
    results = []
    for row in cycles.itertuples(index=False):
        start, end = row.start_index, row.end_index
        if (not np.isfinite(start) or not np.isfinite(end)
                or int(start) != start or int(end) != end
                or not 0 <= start < end <= len(data)):
            raise ValueError(f'周期 {row.cycle_id} 的边界不合法: {start}, {end}')
        features = extract_features(data.iloc[int(start):int(end)], config)
        state, reason, short_score = classify_with_evidence(features, config)
        features.update({
            'cycle_id': row.cycle_id, 'start_index': int(start), 'end_index': int(end),
            'state': state,
            'sub_state': {'effective_discharge': 'normal', 'abnormal_discharge': 'abnormal'}.get(state, 'none'),
            'classification_reason': reason, 'short_score': short_score, 'review_required': True,
        })
        results.append(features)
    if results:
        result = pd.DataFrame(results)
    else:
        result = pd.DataFrame(columns=['cycle_id', 'start_index', 'end_index', 'state', 'sub_state',
                                       'classification_reason', 'short_score', 'review_required', 'data_quality'])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / '放电状态结果.csv', index=False, encoding='utf-8-sig')
    (output / '识别模型参数.json').write_text(
        json.dumps({'model_version': MODEL_VERSION, 'config': asdict(config),
                    'labels_validated': False}, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\n状态统计（自动筛查，尚需人工复核）:')
    print(result['state'].value_counts().reindex(STATES, fill_value=0))
    return result

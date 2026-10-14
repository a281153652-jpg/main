"""噪声自适应的物理状态筛查，输出均需人工复核，非已训练的机器学习模型。"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.ndimage import median_filter

STATES = ('open_circuit', 'effective_discharge', 'short_circuit', 'abnormal_discharge')
MODEL_VERSION = 'dynamic-screening-v4'
DECISION_RULE_VERSION = 'dynamic-decision-v4.1'


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


def _resolution_model(values, windows):
    """估计重复电平网格，不把观测网格间距当作已知硬件 ADC 分辨率。

    对相容的离散网格采用均匀量化误差尺度 q/sqrt(12) 作为保守界。
    两个电平不足以估计 q；大幅真实阶跃不能据此被当成噪声。
    """
    mad_sigma = _noise_sigma(values, windows)
    levels, counts = np.unique(values, return_counts=True)
    step = 0.0
    floor = np.finfo(float).eps * max(1., float(np.max(np.abs(values)))) * 32
    if 3 <= len(levels) <= len(values) // 2:
        gaps = np.diff(levels)
        gaps = gaps[gaps > floor]
        if len(gaps) >= 2:
            candidate = float(np.min(gaps))
            multiples = gaps / candidate
            if np.all(np.abs(multiples - np.rint(multiples)) <= 1e-6):
                step = candidate
    scale = max(mad_sigma, step / np.sqrt(12))
    if step and scale > mad_sigma:
        model = 'quantized_resolution_bound'
    elif mad_sigma <= floor:
        model = 'numerical_floor'
    else:
        model = 'measured_mad'
    baseline = float(levels[np.argmax(counts)]) if step else float(np.median(values))
    return scale, mad_sigma, step, model, baseline


def _runs(mask):
    changes = np.diff(np.r_[False, mask, False].astype(int))
    return np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)


def _event_metrics(raw, filtered, voltage, baseline, sigma_i, mad_sigma_i, step, sigma_u, config):
    """区分事件持续时间和转折点分布；基线只用于形态，不校正测量零点。"""
    n = len(raw)
    guard = max(1, n // 20)
    raw_limit = max(config.noise_multiplier * mad_sigma_i, step / 2)
    raw_mask = np.abs(raw - baseline) > raw_limit
    starts, ends = _runs(raw_mask)
    lengths = ends - starts
    significant = np.abs(filtered - baseline) > config.noise_multiplier * sigma_i
    event_starts, event_ends = _runs(significant)
    # 仅合并过滤窗口范围内的短暂过零，不把相隔很远的尖峰连接成放电段。
    merged = []
    for start, end in zip(event_starts, event_ends):
        if merged and start - merged[-1][1] <= config.median_window:
            merged[-1][1] = int(end)
        else:
            merged.append([int(start), int(end)])
    bounded = [(start, end) for start, end in merged if start >= guard and end <= n - guard]
    span = max((end - start for start, end in bounded), default=0) / n
    u_reference = float(np.quantile(np.abs(voltage), .9))
    response = u_reference - np.abs(voltage) > config.noise_multiplier * sigma_u
    paired = float(np.mean((significant & response)[guard:-guard]))
    return {
        'baseline_current_A': baseline,
        'raw_event_count': len(starts),
        'raw_event_fraction': float(np.mean(raw_mask)),
        'narrow_event_count': int(np.count_nonzero(lengths <= config.median_window)),
        'max_event_width_samples': int(np.max(lengths)) if len(lengths) else 0,
        'oscillation_support_fraction': float(np.mean(significant[guard:-guard])),
        'bounded_excursion_span_fraction': span,
        'paired_response_fraction': paired,
        'pulse_response_supported': (span >= config.oscillation_persistence
                                     and paired >= config.oscillation_persistence),
    }


def _edge_excursion_metrics(current, voltage, sigma_i, sigma_u, config):
    """以两端稳定区形成背景包络；背景不同不等于零点漂移校正。"""
    n = len(current)
    guard = max(1, n // 20)
    left, right = current[:guard], current[-guard:]
    pre, post = float(np.median(left)), float(np.median(right))
    pre_std, post_std = float(np.std(left)), float(np.std(right))
    stable = (guard >= config.min_samples
              and pre_std <= config.noise_multiplier * sigma_i
              and post_std <= config.noise_multiplier * sigma_i)
    lower, upper = min(pre, post), max(pre, post)
    limit = config.noise_multiplier * sigma_i
    outside = (current < lower - limit) | (current > upper + limit)
    starts, ends = _runs(outside)
    merged = []
    for start, end in zip(starts, ends):
        if merged and start - merged[-1][1] <= config.median_window:
            merged[-1][1] = int(end)
        else:
            merged.append([int(start), int(end)])
    bounded = [(start, end) for start, end in merged if start >= guard and end <= n - guard]
    reference = float(np.quantile(np.abs(voltage), .9))
    voltage_response = reference - np.abs(voltage) > config.noise_multiplier * sigma_u
    # 每个完整事件分别查联合支持；最长事件无响应时不能遮蔽另一个有效候选。
    overlaps = np.r_[0, np.cumsum(outside & voltage_response)]
    candidates = [(start, end, (end - start) / n,
                   float((overlaps[end] - overlaps[start]) / (n - 2 * guard)))
                  for start, end in bounded]
    qualified = [event for event in candidates
                 if event[2] >= config.oscillation_persistence
                 and event[3] >= config.oscillation_persistence]
    start, end, span, paired = max(qualified or candidates,
                                    key=lambda event: (event[2], event[3]), default=(0, 0, 0., 0.))
    return {
        'edge_pre_current_A': pre, 'edge_post_current_A': post,
        'edge_pre_std_A': pre_std, 'edge_post_std_A': post_std,
        'edge_reference_sample_count': guard, 'edge_reference_stable': stable,
        'edge_candidate_event_count': len(candidates),
        'edge_supported_event_count': len(qualified) if stable else 0,
        'edge_excursion_start_index': start, 'edge_excursion_end_index': end,
        'edge_excursion_span_fraction': span,
        'edge_paired_response_fraction': paired,
        'edge_pulse_response_supported': stable and bool(qualified),
    }


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
    before = current[guard:max(guard, split - guard)]
    after = current[split + guard:n - guard]
    u_before = np.abs(voltage[guard:max(guard, split - guard)])
    u_after = np.abs(voltage[split + guard:n - guard])
    pre = float(np.median(before)) if len(before) else np.nan
    post = float(np.median(after)) if len(after) else np.nan
    u_pre = float(np.median(u_before)) if len(u_before) else np.nan
    u_post = float(np.median(u_after)) if len(u_after) else np.nan
    valid_pre = len(before) >= config.min_samples
    valid_post = len(after) >= config.min_samples
    supported = valid_pre and valid_post and len(after) / n >= config.min_plateau_fraction
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
        'pre_sample_count': len(before), 'post_sample_count': len(after),
        'plateau_supported': supported,
        'post_current_std': float(np.std(after)) if valid_post else np.nan,
        'post_current_snr': abs(post) / sigma_i if valid_post else np.nan,
        'post_stability_ratio': (float(np.mean(np.abs(after - post) <= config.noise_multiplier * sigma_i))
                                 if valid_post else np.nan),
        'voltage_drop_snr': (u_pre - u_post) / sigma_u,
        'voltage_collapse_ratio': u_post / max(u_pre, sigma_u),
        'post_voltage_std': float(np.std(u_after)) if valid_post else np.nan,
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
    sigma_i, mad_i, level_step_i, noise_model_i, baseline_i = _resolution_model(current, config.analysis_windows)
    sigma_u, mad_u, level_step_u, noise_model_u, _ = _resolution_model(voltage, config.analysis_windows)
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
        'noise_current_mad': mad_i, 'current_level_step_A': level_step_i,
        'current_noise_model': noise_model_i, 'noise_voltage_mad': mad_u,
        'voltage_level_step_V': level_step_u, 'voltage_noise_model': noise_model_u,
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
    features.update(_event_metrics(current, filtered, filtered_u, baseline_i,
                                  sigma_i, mad_i, level_step_i, sigma_u, config))
    features.update(_edge_excursion_metrics(filtered, filtered_u, sigma_i, sigma_u, config))
    features.update(_step_metrics(filtered, filtered_u, sigma_i, sigma_u, config))
    features['current_scale_reliable'] = not (noise_model_i == 'numerical_floor'
                                               and features['raw_event_count'] > 0)
    if not features['current_scale_reliable']:
        # 舍入下限不能作为测量噪声来产生巨大 SNR；保留原尺度及形态以便核查。
        for name in ('activity_snr', 'jump_snr', 'post_current_snr'):
            features[name] = np.nan
    return features


def classify_with_evidence(features, config=None):
    """返回四类建议、原因、短路条件满足比例；分数不是校准概率。"""
    config = config or DynamicConfig()
    if features.get('model_version') != MODEL_VERSION:
        raise ValueError('特征模型版本不匹配，必须从原始波形重新提取特征')
    if features['data_quality'] != 'ok':
        return 'abnormal_discharge', features['data_quality'], 0.0
    if not features['current_scale_reliable']:
        return 'abnormal_discharge', 'unreliable_current_noise_scale', 0.0
    z = config.noise_multiplier
    oscillatory = (features['reversal_count'] >= config.min_reversals
                   and features['oscillation_index'] >= config.oscillation_persistence
                   and features['oscillation_span_ratio'] >= config.oscillation_persistence
                   and features['oscillation_support_fraction'] >= config.oscillation_persistence)
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
    short_score = (sum(bool(v) for v in evidence) / len(evidence)
                   if features['plateau_supported'] else 0.0)
    if features['plateau_supported'] and all(evidence):
        return 'short_circuit', 'joint_step_then_stable_plateau', short_score
    if oscillatory and features['voltage_response_snr'] > z:
        return 'effective_discharge', 'sustained_oscillation_with_voltage_response', short_score
    if features['pulse_response_supported'] and features['voltage_response_snr'] > z:
        return 'effective_discharge', 'coherent_current_excursion_with_voltage_response', short_score
    if features['edge_pulse_response_supported'] and features['voltage_response_snr'] > z:
        return 'effective_discharge', 'complete_excursion_outside_edge_backgrounds', short_score
    # 全周期 RMS 随占空比降低，不能否决已取得局部幅度及联合时间支持的事件。
    # 保留原幅度/持续时间条件；无联合支持时才使用整体活动作背景尺度判断。
    if features['activity_snr'] <= z:
        if features['current_noise_model'] == 'quantized_resolution_bound' and features['raw_event_count']:
            return 'abnormal_discharge', 'resolution_limited_events', 0.0
        if features['voltage_snr'] > z:
            return 'open_circuit', 'current_at_noise_floor', 0.0
        return 'abnormal_discharge', 'no_voltage_drive_evidence', 0.0
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
            'decision_rule_version': DECISION_RULE_VERSION,
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
        json.dumps({'model_version': MODEL_VERSION, 'decision_rule_version': DECISION_RULE_VERSION,
                    'config': asdict(config),
                    'labels_validated': False}, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\n状态统计（自动筛查，尚需人工复核）:')
    print(result['state'].value_counts().reindex(STATES, fill_value=0))
    print('模型版本:', MODEL_VERSION)
    if 'current_noise_model' in result:
        print('电流噪声/分辨尺度模型:', result['current_noise_model'].value_counts().to_dict())
        print('没有足够前后平台证据的周期:', int(result['plateau_supported'].ne(True).sum()))
    return result

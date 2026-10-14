"""只收集待定周期的诊断证据，不重新分类或训练状态模型。"""
from dataclasses import asdict

import numpy as np
import pandas as pd

from modules.放电状态识别 import DynamicConfig, STATES


FEATURE_VERSION = 'dynamic-screening-v4'
GROUP_COLUMNS = ['classification_reason', 'morphology_cluster', 'batch_reason']
REQUIRED = ['model_version', 'cycle_id', 'automatic_state', *GROUP_COLUMNS]
SUMMARY_FEATURES = [
    'raw_longest_event_fraction', 'narrow_event_ratio', 'activity_snr',
    'oscillation_support_fraction', 'bounded_excursion_span_fraction',
    'edge_excursion_span_fraction', 'edge_paired_response_fraction',
    'current_levelstep_to_scale', 'current_level_step_A', 'noise_current',
    'joint_event_behind_low_activity_gate',
]


def validate_results(results):
    missing = sorted(set(REQUIRED) - set(results.columns))
    if missing:
        raise ValueError(f'自学习结果缺少必需列: {missing}')
    if results.empty or not results['model_version'].eq(FEATURE_VERSION).all():
        raise ValueError('需要非空、同一 dynamic-screening-v4 版本的自学习状态结果.csv')
    if not results['automatic_state'].isin([*STATES, 'undetermined']).all():
        raise ValueError('automatic_state 含缺失或未支持的状态，无法可靠统计待定周期')
    ids = pd.to_numeric(results['cycle_id'], errors='coerce')
    values = ids.to_numpy(dtype=float)
    if (not np.isfinite(values).all() or not (values > 0).all()
            or not (values == np.floor(values)).all() or ids.duplicated().any()):
        raise ValueError('cycle_id 必须为唯一的有限正整数')


def _numeric(frame, column):
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    values = pd.to_numeric(frame[column], errors='coerce').astype(float)
    return values.where(np.isfinite(values))


def _unreliable_scale(frame):
    bad = pd.Series(False, index=frame.index)
    if 'current_scale_reliable' in frame:
        # 缺失可靠性声明不能作为肯定的尺度证据。
        flags = frame['current_scale_reliable'].astype('string').str.lower()
        bad |= flags.ne('true').fillna(True)
    if 'current_noise_model' in frame:
        bad |= frame['current_noise_model'].eq('numerical_floor')
    return bad


def _true(frame, column):
    if column not in frame:
        return pd.Series(False, index=frame.index)
    return frame[column].astype('string').str.lower().eq('true').fillna(False)


def _joint_event_gate_flag(frame):
    """按当前默认参考配置定位前置活动门控遮蔽，不据此改变状态。"""
    z = DynamicConfig().noise_multiplier
    quality = (frame['data_quality'].eq('ok') if 'data_quality' in frame
               else pd.Series(False, index=frame.index))
    joint_event = _true(frame, 'pulse_response_supported') | _true(frame, 'edge_pulse_response_supported')
    return (quality & _true(frame, 'current_scale_reliable') & ~_unreliable_scale(frame)
            & _numeric(frame, 'activity_snr').le(z)
            & _numeric(frame, 'voltage_response_snr').gt(z) & joint_event)


def diagnostic_features(frame):
    """派生比例中的缺失、零分母和不可信尺度均保留为 NaN。"""
    values = pd.DataFrame(index=frame.index)
    samples = _numeric(frame, 'sample_count').where(lambda v: v > 0)
    events = _numeric(frame, 'raw_event_count').where(lambda v: v > 0)
    widths = _numeric(frame, 'max_event_width_samples').where(lambda v: v >= 0)
    narrow = _numeric(frame, 'narrow_event_count').where(lambda v: v >= 0)
    values['raw_longest_event_fraction'] = (widths / samples).where(widths <= samples)
    values['narrow_event_ratio'] = (narrow / events).where(narrow <= events)
    for name in SUMMARY_FEATURES[2:7]:
        values[name] = _numeric(frame, name)
    unreliable = _unreliable_scale(frame)
    values['activity_snr'] = values['activity_snr'].where(~unreliable)
    noise = _numeric(frame, 'noise_current').where(lambda v: v > 0)
    step = _numeric(frame, 'current_level_step_A').where(lambda v: v > 0)
    values['current_levelstep_to_scale'] = (step / noise).where(~unreliable)
    values['current_level_step_A'] = _numeric(frame, 'current_level_step_A')
    values['noise_current'] = _numeric(frame, 'noise_current')
    values['joint_event_behind_low_activity_gate'] = _joint_event_gate_flag(frame)
    return values.reindex(columns=SUMMARY_FEATURES).replace([np.inf, -np.inf], np.nan)


def _json_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _group_record(key):
    return {name: _json_value(value) for name, value in zip(GROUP_COLUMNS, key)}


def _key_order(key):
    # 显式区分缺失值，避免它们在默认 groupby 中被丢弃。
    return tuple((0, '') if pd.isna(value) else (1, str(value)) for value in key)


def _closest(subset, features):
    distance = _numeric(subset, 'cluster_distance_squared')
    finite = distance.dropna()
    if len(finite):
        eligible = subset.loc[finite.index].copy()
        eligible['_diagnostic_distance'] = finite
        row = eligible.sort_values(['_diagnostic_distance', 'cycle_id'], kind='stable').iloc[0]
        return row.name, 'closest_to_group_cluster_center'
    span = features.loc[subset.index, 'raw_longest_event_fraction'].dropna()
    if len(span):
        eligible = subset.loc[span.index].copy()
        eligible['_diagnostic_median_distance'] = np.abs(span - span.median())
        row = eligible.sort_values(['_diagnostic_median_distance', 'cycle_id'], kind='stable').iloc[0]
        return row.name, 'raw_longest_span_nearest_group_median_no_finite_cluster_distance'
    return subset.sort_values('cycle_id', kind='stable').index[0], 'lowest_cycle_id_no_finite_diagnostic_distance_or_span'


def _group_schedule(groups):
    """有限预算先保留尺度未知/离群，再覆盖原因，再覆盖各个原因群组。"""
    ordered = sorted(groups, key=lambda group: (-len(group['rows']), _key_order(group['key'])))
    queue, used, reasons = [], set(), set()

    def add(group):
        if group['number'] not in used:
            queue.append(group)
            used.add(group['number'])
            reasons.add(_key_order((group['key'][0],)))

    for group in ordered:
        reason, _, batch = group['key']
        if reason == 'unreliable_current_noise_scale' or batch == 'untrusted_numerical_noise_floor':
            add(group)
    for group in ordered:
        if isinstance(group['key'][2], str) and 'outlier' in group['key'][2]:
            add(group)
    for group in ordered:
        if _key_order((group['key'][0],)) not in reasons:
            add(group)
    for group in ordered:
        add(group)
    return queue


def select_representatives(pending, features, examples):
    if not isinstance(examples, int) or isinstance(examples, bool) or examples <= 0:
        raise ValueError('examples 必须为正整数')
    groups = [
        {'number': number, 'key': key, 'rows': subset}
        for number, (key, subset) in enumerate(pending.groupby(GROUP_COLUMNS, dropna=False, sort=False))
    ]
    selected, selected_indices, selected_groups = [], set(), set()
    schedule = _group_schedule(groups)
    gated = features['joint_event_behind_low_activity_gate'].eq(True)
    if gated.any():
        index, _ = _closest(pending.loc[gated], features)
        selected.append((index, 'joint_event_behind_low_activity_gate'))
        selected_indices.add(index)
        selected_groups.update(group['number'] for group in groups if index in group['rows'].index)
    for group in schedule:
        if len(selected) >= examples:
            break
        index, reason = _closest(group['rows'], features)
        if index in selected_indices:
            selected_groups.add(group['number'])
            continue
        selected.append((index, reason))
        selected_indices.add(index)
        selected_groups.add(group['number'])
    # 已覆盖所有可覆盖小组后，按小组大小轮流补选各组最长原始事件。
    supplemental = sorted(groups, key=lambda group: (-len(group['rows']), _key_order(group['key'])))
    while len(selected) < min(examples, len(pending)):
        added = False
        for group in supplemental:
            remaining = group['rows'].loc[~group['rows'].index.isin(selected_indices)].copy()
            if remaining.empty:
                continue
            remaining['_diagnostic_span'] = features.loc[remaining.index, 'raw_longest_event_fraction']
            index = remaining.sort_values(['_diagnostic_span', 'cycle_id'], ascending=[False, True],
                                           na_position='last', kind='stable').index[0]
            reason = ('longest_raw_event_span_additional_group_sample'
                      if np.isfinite(features.loc[index, 'raw_longest_event_fraction'])
                      else 'lowest_cycle_id_additional_sample_no_finite_raw_span')
            selected.append((index, reason))
            selected_indices.add(index)
            selected_groups.add(group['number'])
            added = True
            if len(selected) >= examples:
                break
        if not added:
            break
    representatives = pending.loc[[index for index, _ in selected]].copy()
    # 原有 state/automatic_state 只是来源记录，保留其值，不重写或推断状态。
    representatives['selection_reason'] = [reason for _, reason in selected]
    for name in features:
        if name not in representatives:
            representatives[name] = features.loc[representatives.index, name]
    uncovered = [{**_group_record(group['key']), 'cycles': len(group['rows'])}
                 for group in schedule if group['number'] not in selected_groups]
    return representatives, len(groups), len(selected_groups), uncovered


def diagnose_pending(results, examples=12):
    """返回诊断表与说明；输入 DataFrame 和所有状态列保持不变。"""
    validate_results(results)
    work = results.copy(deep=True).reset_index(drop=True)
    work['cycle_id'] = pd.to_numeric(work['cycle_id']).astype('int64')
    pending = work.loc[work['automatic_state'].eq('undetermined')].copy()
    counts = pending.groupby(GROUP_COLUMNS, dropna=False).size().rename('cycles').reset_index()
    counts = counts.sort_values('cycles', ascending=False, kind='stable').reset_index(drop=True)
    features = diagnostic_features(pending)
    summary_rows = []
    for key, subset in pending.groupby(GROUP_COLUMNS, dropna=False, sort=False):
        for name in SUMMARY_FEATURES:
            finite = features.loc[subset.index, name].astype(float).dropna()
            record = {**_group_record(key), 'cycles': len(subset), 'feature': name,
                      'finite_count': len(finite), 'missing_or_nonfinite_count': len(subset) - len(finite)}
            record.update({label: float(finite.quantile(q)) if len(finite) else np.nan
                           for label, q in [('min', 0), ('p10', .1), ('p50', .5), ('p90', .9), ('max', 1)]})
            summary_rows.append(record)
    summary = pd.DataFrame(summary_rows, columns=[*GROUP_COLUMNS, 'cycles', 'feature',
                                                  'finite_count', 'missing_or_nonfinite_count',
                                                  'min', 'p10', 'p50', 'p90', 'max'])
    representatives, group_count, covered_count, uncovered = select_representatives(pending, features, examples)
    reason_counts = [
        {'classification_reason': _json_value(reason), 'cycles': int(count)}
        for reason, count in pending.groupby('classification_reason', dropna=False).size().items()
    ]
    metadata = {
        'diagnostic_version': 'pending-diagnostics-v1', 'feature_model_version': FEATURE_VERSION,
        'total_cycles': len(work), 'pending_cycles': len(pending),
        'nonpending_cycles': len(work) - len(pending), 'classification_reason_counts': reason_counts,
        'requested_examples': examples, 'selected_cycles': representatives['cycle_id'].tolist(),
        'selected_count': len(representatives), 'pending_reason_groups': group_count,
        'covered_reason_groups': covered_count, 'uncovered_reason_groups': uncovered,
        'labels_changed': False, 'labels_validated': False,
        'joint_event_behind_low_activity_gate_cycles': int(
            features['joint_event_behind_low_activity_gate'].sum()),
        'default_dynamic_config_reference': asdict(DynamicConfig()),
        'config_reference_note': '当前程序默认配置仅供解释特征；未核实原批次使用的配置，不执行重分类。',
        'feature_notes': {
            'raw_longest_event_fraction': 'max_event_width_samples / sample_count；描述原始连续事件宽度。',
            'narrow_event_ratio': 'narrow_event_count / raw_event_count；没有事件时比例无定义，留空。',
            'current_levelstep_to_scale': 'current_level_step_A / noise_current；无有效网格或不可信尺度时留空。',
            'activity_snr': '动态特征尺度比，不是机器学习概率；不可信尺度不参与该摘要。',
            'quantiles': '仅使用有限数值；缺失或非有限值单独计数，不以零填充。',
            'resolution_limited_events': '当前尺度和形态证据不足；不能直接判作开路、噪声或异常加工。',
            'selection': '先按原因/形态群组/批次原因覆盖代表，再补选各组最长原始事件；不重复。',
            'joint_event_behind_low_activity_gate': (
                '按当前默认 z=4 参考定位：数据质量 ok、尺度可信、activity_snr<=z、'
                'voltage_response_snr>z，且脉冲响应或两端包络脉冲响应支持为真。'
                '该只读标志描述可能被前置门控遮蔽的联合事件；不是状态真值。'
                '如存在则优先选至少一个代表，再覆盖原因组。'),
        },
    }
    return counts, summary, representatives, metadata

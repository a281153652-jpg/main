"""按已记录的配置重判 v4 特征并重做批次学习；不读取原始波形。"""
import argparse
from dataclasses import asdict, fields
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import uuid

import numpy as np
import pandas as pd

from modules.放电状态识别 import (
    DECISION_RULE_VERSION, MODEL_VERSION, STATES, DynamicConfig, classify_with_evidence,
)
from modules.批次自学习 import REQUIRED as BATCH_REQUIRED, learn_batch_states
from modules.判别证据审计 import write_decision_audit


BOOLEAN_FEATURES = (
    'current_scale_reliable', 'plateau_supported', 'pulse_response_supported',
    'edge_pulse_response_supported', 'edge_reference_stable',
)
NUMERIC_FEATURES = (
    'noise_current', 'noise_voltage', 'activity_snr', 'voltage_snr',
    'reversal_count', 'oscillation_index', 'oscillation_span_ratio',
    'oscillation_support_fraction', 'raw_event_count', 'step_bic_gain',
    'jump_snr', 'transition_fraction', 'plateau_fraction', 'post_current_snr',
    'post_current_std', 'voltage_drop_snr', 'voltage_collapse_ratio',
    'post_voltage_std', 'voltage_response_snr', 'robust_I_pp',
    'raw_event_fraction', 'bounded_excursion_span_fraction',
    'paired_response_fraction', 'max_event_width_samples', 'post_stability_ratio',
    'edge_excursion_span_fraction', 'edge_paired_response_fraction',
    'current_zero_offset_A',
)
LEARNING_COLUMNS = (
    'batch_model_version', 'morphology_cluster', 'cluster_membership',
    'cluster_distance_squared', 'distribution_outlier', 'automatic_state',
    'decision_status', 'review_priority', 'batch_reason',
    'noise_scale_reliable_for_learning', 'batch_learning_status',
    'dynamic_review_required', 'automatic_label_validated',
)


def _integer_column(frame, name, minimum):
    value = pd.to_numeric(frame[name], errors='raise')
    if (not np.isfinite(value).all() or not value.ge(minimum).all()
            or not value.mod(1).eq(0).all()
            or not value.le(np.iinfo(np.int64).max).all()):
        raise ValueError(f'{name} 必须为不小于 {minimum} 的有限整数')
    frame[name] = value.astype('int64')


def _load_config(path):
    if not path.is_file():
        raise ValueError(f'找不到已记录的识别配置: {path}；请提供 --model-config')
    def reject_constant(value):
        raise ValueError(f'识别配置含非有限 JSON 数值: {value}')
    recorded = json.loads(path.read_text(encoding='utf-8-sig'), parse_constant=reject_constant)
    if not isinstance(recorded, dict) or recorded.get('model_version') != MODEL_VERSION:
        raise ValueError('识别配置必须明确记录 dynamic-screening-v4 特征版本')
    values = recorded.get('config')
    expected = {field.name for field in fields(DynamicConfig)}
    if not isinstance(values, dict) or set(values) != expected:
        raise ValueError('识别配置必须完整记录 DynamicConfig 字段，不能使用缺失字段的默认值')
    try:
        config = DynamicConfig(**values)
    except (TypeError, ValueError) as error:
        raise ValueError(f'已记录的 DynamicConfig 不合法: {error}') from error
    # 配置中需用于切片/计数的项不能以浮点或布尔值冒充整数。
    for name in ('median_window', 'analysis_windows', 'min_samples', 'min_reversals'):
        if type(values[name]) is not int:
            raise ValueError(f'识别配置 {name} 必须为整数')
    for name in expected - {'median_window', 'analysis_windows', 'min_samples', 'min_reversals'}:
        if type(values[name]) not in (int, float):
            raise ValueError(f'识别配置 {name} 必须为数值')
    return config, recorded


def _validate_frame(frame, config):
    required = set(BATCH_REQUIRED) | set(BOOLEAN_FEATURES) | set(NUMERIC_FEATURES) | {
        'start_index', 'end_index', 'current_noise_model',
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f'v4 特征表缺少列: {sorted(missing)}')
    if frame.empty:
        raise ValueError('特征表没有周期记录')
    if not frame['model_version'].eq(MODEL_VERSION).all():
        raise ValueError('只支持完整的同一 v4 特征；不接受 v3、混合版本或缺失版本')
    for name, minimum in (('cycle_id', 1), ('start_index', 0), ('end_index', 1), ('sample_count', 1)):
        _integer_column(frame, name, minimum)
    if frame['cycle_id'].duplicated().any():
        raise ValueError('cycle_id 必须唯一')
    if not (frame['end_index'] - frame['start_index']).eq(frame['sample_count']).all():
        raise ValueError('原周期边界必须为左闭右开，且 end_index-start_index 等于 sample_count')
    for name in ('data_quality', 'classification_reason'):
        if frame[name].isna().any() or not frame[name].map(lambda value: isinstance(value, str) and bool(value)).all():
            raise ValueError(f'{name} 必须记录非空文本')
    if not frame['state'].isin(STATES).all():
        raise ValueError('包含不支持的原始动态状态')
    for name in NUMERIC_FEATURES:
        values = pd.to_numeric(frame[name], errors='raise')
        if np.isinf(values).any():
            raise ValueError(f'{name} 含无限值')
        frame[name] = values
    if not frame['current_zero_offset_A'].eq(config.current_zero_offset).all():
        raise ValueError('特征表记录的电流零点与识别配置不一致，不能按不同配置重判')
    good = frame['data_quality'].eq('ok')
    for name in BOOLEAN_FEATURES:
        # CSV 中坏数据行可能是空值，正常行必须有明确的布尔证据。
        values = frame[name]
        valid = values.map(lambda value: isinstance(value, (bool, np.bool_)))
        text_boolean = values.isin(('True', 'False'))
        if not (valid | text_boolean | ~good).all():
            raise ValueError(f'{name} 的有效数据行必须为 True 或 False')
        if text_boolean.any():
            frame[name] = values.map(lambda value: {'True': True, 'False': False}.get(value, value)
                                     if isinstance(value, str) else value)
    trusted = good & frame['current_scale_reliable'].eq(True)
    finite_required = ('noise_current', 'noise_voltage', 'activity_snr', 'voltage_snr',
                       'voltage_response_snr', 'reversal_count', 'oscillation_index',
                       'oscillation_span_ratio', 'oscillation_support_fraction', 'raw_event_count')
    for name in finite_required:
        if not np.isfinite(frame.loc[trusted, name]).all():
            raise ValueError(f'{name} 的可信尺度有效行缺少有限值')
    if not frame.loc[trusted, ['noise_current', 'noise_voltage']].gt(0).all().all():
        raise ValueError('可信尺度有效行的电流/电压尺度必须为正数')


def _counts(series):
    return {str(name): int(count) for name, count in series.value_counts().items()}


def reclassify_features(feature_file, output_dir, model_config=None):
    path = Path(feature_file)
    if not path.is_file():
        raise ValueError(f'找不到特征表: {path}')
    config_path = Path(model_config) if model_config else path.parent / '识别模型参数.json'
    config, recorded = _load_config(config_path)
    snapshot = path.read_bytes()
    original = pd.read_csv(BytesIO(snapshot), encoding='utf-8-sig', float_precision='round_trip')
    work = original.copy(deep=True)
    _validate_frame(work, config)
    prior_rule = (original['decision_rule_version'].fillna('not_recorded').astype(str)
                  if 'decision_rule_version' in original else pd.Series('not_recorded', index=original.index))
    renamed = {}
    for name in LEARNING_COLUMNS:
        if name in work:
            target = 'previous_learning_' + name
            while target in work or target in renamed.values():
                target = 'previous_' + target
            renamed[name] = target
    work = work.rename(columns=renamed)
    decisions = [classify_with_evidence(row, config) for row in work.to_dict(orient='records')]
    work['state'] = [value[0] for value in decisions]
    work['classification_reason'] = [value[1] for value in decisions]
    work['short_score'] = [value[2] for value in decisions]
    work['sub_state'] = work['state'].map({
        'effective_discharge': 'normal', 'abnormal_discharge': 'abnormal',
        'open_circuit': 'none', 'short_circuit': 'none',
    })
    work['review_required'] = True
    work['decision_rule_version'] = DECISION_RULE_VERSION
    # 到这里配置、特征及逐行判别全部验证完毕；此前不创建任何输出。
    folder = Path(output_dir) / (
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    work.to_csv(folder / '放电状态结果.csv', index=False, encoding='utf-8-sig')
    updated_config = {**recorded, 'config': asdict(config), 'model_version': MODEL_VERSION,
                      'decision_rule_version': DECISION_RULE_VERSION, 'labels_validated': False}
    (folder / '识别模型参数.json').write_text(
        json.dumps(updated_config, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    provenance = {
        'input_filename': path.name, 'input_feature_sha256': hashlib.sha256(snapshot).hexdigest(),
        'input_feature_file': str(path.resolve()), 'model_config_file': str(config_path.resolve()),
        'previous_rule_versions': sorted(prior_rule.unique().tolist()),
        'fresh_rule': DECISION_RULE_VERSION, 'feature_version': MODEL_VERSION,
        'computed_from_raw': False, 'cycles': len(work), 'experimental_labels_validated': False,
        'previous_learning_columns': renamed,
    }
    learned = learn_batch_states(work, folder, source_info=provenance)
    # 批次学习默认审计参数；重判应按此次读取的真实配置输出审计。
    write_decision_audit(learned, folder, config)
    comparison = pd.DataFrame({
        'cycle_id': work['cycle_id'], 'previous_state': original['state'],
        'previous_classification_reason': original['classification_reason'],
        'previous_decision_rule_version': prior_rule,
        'new_state': work['state'], 'new_classification_reason': work['classification_reason'],
        'new_decision_rule_version': DECISION_RULE_VERSION,
        'new_automatic_state': learned['automatic_state'],
    })
    comparison['previous_automatic_state'] = (
        original['automatic_state'] if 'automatic_state' in original else pd.Series(None, index=original.index, dtype='object'))
    comparison['state_changed'] = comparison['previous_state'].ne(comparison['new_state'])
    comparison['reason_changed'] = comparison['previous_classification_reason'].ne(comparison['new_classification_reason'])
    comparison['automatic_state_changed'] = (
        comparison['previous_automatic_state'].ne(comparison['new_automatic_state'])
        if 'automatic_state' in original else pd.Series(None, index=original.index, dtype='object'))
    comparison.to_csv(folder / '状态变更对照.csv', index=False, encoding='utf-8-sig')
    provenance.update({
        'dynamic_state_changes': int(comparison['state_changed'].sum()),
        'classification_reason_changes': int(comparison['reason_changed'].sum()),
        'previous_dynamic_counts': _counts(original['state']), 'new_dynamic_counts': _counts(work['state']),
        'previous_automatic_counts': _counts(original['automatic_state']) if 'automatic_state' in original else None,
        'new_automatic_counts': _counts(learned['automatic_state']),
        'automatic_state_changes': int(comparison['automatic_state_changed'].sum()) if 'automatic_state' in original else None,
        'classification_reason_transitions': [
            {'previous_reason': before, 'new_reason': after, 'cycles': int(count)}
            for (before, after), count in comparison.groupby(
                ['previous_classification_reason', 'new_classification_reason']).size().items()
        ],
        'missing_previous_automatic_state_note': (
            None if 'automatic_state' in original else '输入为动态结果表，未记录旧批次自动状态，旧自动数量与变更数未推算'),
    })
    (folder / '运行来源.json').write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'\n动态候选状态变化: {provenance["dynamic_state_changes"]} / {len(work)}')
    print(f'判别原因变化: {provenance["classification_reason_changes"]} / {len(work)}')
    print('本次只重判已有特征，候选标签尚未独立验证。')
    print('独立结果目录:', folder.resolve())
    return learned, folder


def main(argv=None):
    parser = argparse.ArgumentParser(description='按已记录的 v4 配置重判特征并重做批次学习，不读取波形')
    parser.add_argument('feature_file', help='已有 v4 放电状态结果.csv（也可提供完整 v4 自学习结果）')
    parser.add_argument('--output', default='../输出结果/动态判别复核_v4_1', help='独立结果根目录，每次创建新的子目录')
    parser.add_argument('--model-config', help='识别模型参数.json；默认使用输入表同目录的文件')
    args = parser.parse_args(argv)
    try:
        return reclassify_features(args.feature_file, args.output, args.model_config)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()

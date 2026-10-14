"""只读审计局部事件分段预算敏感性；支持次数不是状态概率或准确率。"""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import time
from uuid import uuid4

import numpy as np
import pandas as pd

from modules.data_loader import load_waveform_data
from modules.局部事件模型 import EVENT_MODEL_VERSION, LocalEventConfig, fit_local_events
from modules.放电状态识别 import MODEL_VERSION, STATES, extract_features
from refine_waveform_events import (
    _boolean, _fingerprint, _integer_column, _load_config, _match_features,
    _sha256, _snapshot, _unchanged, _validate_frame, _waveform_valid,
)


BUDGETS = (4, 6, 8, 10)
BASELINE_BUDGET = 6
AUDIT_VERSION = 'local-event-segment-budget-audit-v1'
BOOLEAN_COLUMNS = (
    'new_automatic_label_validated', 'local_event_state_changed',
    'local_event_automatic_changed', 'event_supported', 'event_boundary_truncated',
    'event_scale_reliable', 'event_structure_score_is_probability',
    'event_supported_count_is_proposal_count', 'event_boundary_flag_is_proven_truncation',
)
OPTIONAL_BOOLEAN_COLUMNS = (
    'event_segment_cap_hit', 'event_boundary_grid_capped', 'event_voltage_response_supported',
)
SUMMARY_COLUMNS = (
    'stability_supported_budgets', 'stability_supported_budget_count',
    'stability_supported_all_budgets', 'stability_supported_any_budget',
    'stability_supported_start_spread_samples', 'stability_supported_end_spread_samples',
    'stability_baseline_replay_matches', 'stability_review_priority_suggestion',
    'stability_is_state_probability', 'stability_is_experimental_validation',
)


def _canonical_empty(value):
    """空拒绝原因在 CSV 中可变成 NaN；不把一般非空文本转为缺失。"""
    return value is None or (isinstance(value, str) and value == '') or bool(pd.isna(value))


def _event_evidence(event):
    if not isinstance(event, dict) or not {'event_supported', 'event_reason'} <= set(event):
        raise ValueError('事件模型必须返回 event_supported 和 event_reason 的平坦字典')
    for name, value in event.items():
        if not isinstance(name, str) or not name.startswith('event_'):
            raise ValueError('事件模型只能返回 event_* 证据列')
        if isinstance(value, (dict, list, tuple, np.ndarray)):
            raise ValueError('事件证据必须为平坦标量')
        if isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)) and np.isinf(value):
            raise ValueError('事件证据不能含无限值')
    event = event.copy()
    event['event_supported'] = _boolean(event['event_supported'], 'event_supported')
    if not isinstance(event['event_reason'], str) or not event['event_reason']:
        raise ValueError('event_reason 必须为非空文本')
    if event.get('event_model_version') != EVENT_MODEL_VERSION:
        raise ValueError('事件模型返回的版本与本审计支持版本不一致')
    return event


def _match_event(recorded, replay, event_columns):
    """重演核对全部现有事件列和新计算列；未返回的稀疏证据应为空。"""
    for name in sorted(set(event_columns) | set(replay)):
        if name not in recorded:
            raise ValueError(f'周期 {recorded["cycle_id"]} 缺少原事件证据: {name}')
        previous, actual = recorded[name], replay.get(name)
        if _canonical_empty(actual):
            equal = _canonical_empty(previous)
        elif isinstance(actual, (bool, np.bool_)):
            equal = _boolean(previous, name) == bool(actual)
        elif isinstance(actual, str):
            equal = previous == actual
        else:
            try:
                equal = bool(np.isclose(float(previous), float(actual), rtol=1e-9, atol=1e-12))
            except (TypeError, ValueError):
                equal = False
        if not equal:
            raise ValueError(f'周期 {recorded["cycle_id"]} 的预算 6 重演与原事件证据不一致: {name}')


def _validate_results(frame, config):
    for name, minimum in (('cycle_id', 1), ('start_index', 0), ('end_index', 1), ('sample_count', 1)):
        if name in frame:
            _integer_column(frame, name, minimum)
    _validate_frame(frame, config)
    required = {'automatic_state', 'new_automatic_state', 'new_dynamic_state',
                'event_model_version', 'event_reason'} | set(BOOLEAN_COLUMNS)
    if required - set(frame):
        raise ValueError(f'只接受已完成局部事件复核的结果表，缺少列: {sorted(required - set(frame))}')
    if not frame['event_model_version'].eq(EVENT_MODEL_VERSION).all():
        raise ValueError(f'只支持同一 {EVENT_MODEL_VERSION} 事件版本')
    for name in ('automatic_state', 'new_automatic_state'):
        if not frame[name].isin((*STATES, 'undetermined')).all():
            raise ValueError(f'{name} 含不支持的状态')
    if not frame['new_dynamic_state'].isin(STATES).all():
        raise ValueError('new_dynamic_state 含不支持的状态')
    if not frame['event_reason'].map(lambda value: isinstance(value, str) and bool(value)).all():
        raise ValueError('event_reason 必须为非空文本')
    for name in BOOLEAN_COLUMNS:
        frame[name] = frame[name].map(lambda value: _boolean(value, name))
    for name in OPTIONAL_BOOLEAN_COLUMNS:
        if name in frame:
            present = frame[name].notna()
            # 保持 object，不把 NaN 与 False 合并。
            frame.loc[present, name] = frame.loc[present, name].map(lambda value: _boolean(value, name))
    if not frame['local_event_state_changed'].eq(frame['new_dynamic_state'].ne(frame['state'])).all():
        raise ValueError('local_event_state_changed 与原/新动态状态不一致')
    if not frame['local_event_automatic_changed'].eq(frame['new_automatic_state'].ne(frame['automatic_state'])).all():
        raise ValueError('local_event_automatic_changed 与原/新自动状态不一致')
    ordered = frame.sort_values('start_index')
    if (ordered['start_index'].to_numpy()[1:] < ordered['end_index'].to_numpy()[:-1]).any():
        raise ValueError('结果表原周期边界存在重叠')
    if set(SUMMARY_COLUMNS) & set(frame):
        raise ValueError('输入已含本审计摘要列，请使用局部事件复核结果表')


def _spread(events, name):
    values = [float(event[name]) for event in events if event['event_supported'] and name in event]
    return max(values) - min(values) if values else np.nan


def audit_event_stability(raw_source, *, results, model_config, expected_source_sha256,
                          output_dir, include_retained=False):
    expected = _fingerprint(expected_source_sha256, '--expected-source-sha256')
    result_path, config_path, source_path = map(Path, (results, model_config, raw_source))
    snapshot, result_hash = _snapshot(result_path)
    _, config_hash = _snapshot(config_path)
    config, recorded_config = _load_config(config_path)
    _unchanged(config_path, config_hash)
    original = pd.read_csv(BytesIO(snapshot), encoding='utf-8-sig', float_precision='round_trip')
    validated = original.copy(deep=True)
    _validate_results(validated, config)
    scope = validated['new_automatic_state'].eq('effective_discharge')
    if not include_retained:
        scope &= validated['automatic_state'].ne('effective_discharge')
    selected = validated.loc[scope]
    # 即使没有新增候选，也实际验证来源哈希；不虚构读取点数或特征核对。
    source_hash = _sha256(source_path)
    if source_hash != expected:
        raise ValueError('原始 CSV SHA-256 与预期不一致；未读取波形或写出结果')
    source_samples = None
    if not selected.empty:
        data = load_waveform_data(source_path)
        _unchanged(source_path, source_hash)
        _waveform_valid(data, '原始数据')
        source_samples = len(data)
        if not validated['end_index'].le(source_samples).all():
            raise ValueError('结果表原周期边界超出原始 CSV 样本数')
    event_columns = [name for name in original if name.startswith('event_')]
    base = LocalEventConfig(noise_multiplier=config.noise_multiplier,
                           min_segment_samples=config.min_samples)
    trial_configs = [replace(base, max_segments=budget) for budget in BUDGETS]
    details, summaries = [], []
    last_progress = time.monotonic()
    for processed, (_, row) in enumerate(selected.iterrows(), start=1):
        block = data.iloc[int(row['start_index']):int(row['end_index'])].copy().reset_index(drop=True)
        features = extract_features(block, config)
        _match_features(row, features)
        # 先重演原预算；出现差异时不能用不同版本/设置悄悄取代原证据。
        baseline = _event_evidence(fit_local_events(block, features, config=base))
        _match_event(row, baseline, event_columns)
        events = []
        for trial_config in trial_configs:
            event = (baseline if trial_config.max_segments == BASELINE_BUDGET else
                     _event_evidence(fit_local_events(block, features, config=trial_config)))
            events.append(event)
            details.append({'cycle_id': int(row['cycle_id']),
                            'start_index': int(row['start_index']), 'end_index': int(row['end_index']),
                            'automatic_state': row['automatic_state'],
                            'new_automatic_state': row['new_automatic_state'],
                            'trial_max_segments': trial_config.max_segments,
                            'trial_noise_multiplier': trial_config.noise_multiplier,
                            'trial_min_segment_samples': trial_config.min_segment_samples,
                            'trial_max_boundary_candidates': trial_config.max_boundary_candidates,
                            **event})
            if time.monotonic() - last_progress >= 20:
                print(f'分段预算稳定性审计进度: {processed}/{len(selected)}，预算 {trial_config.max_segments}', flush=True)
                last_progress = time.monotonic()
        supported = [budget for budget, event in zip(BUDGETS, events) if event['event_supported']]
        preserved = original.loc[original['cycle_id'].eq(row['cycle_id'])].iloc[0].to_dict()
        summaries.append({**preserved, 'stability_supported_budgets': '|'.join(map(str, supported)),
                          'stability_supported_budget_count': len(supported),
                          'stability_supported_all_budgets': len(supported) == len(BUDGETS),
                          'stability_supported_any_budget': bool(supported),
                          'stability_supported_start_spread_samples': _spread(events, 'event_start_index'),
                          'stability_supported_end_spread_samples': _spread(events, 'event_end_index'),
                          'stability_baseline_replay_matches': True,
                          'stability_review_priority_suggestion': ('inspect_segmentation_sensitivity' if 0 < len(supported) < len(BUDGETS)
                                                                  else 'inspect_waveform_evidence'),
                          'stability_is_state_probability': False,
                          'stability_is_experimental_validation': False})
    for path, fingerprint in ((result_path, result_hash), (config_path, config_hash), (source_path, source_hash)):
        _unchanged(path, fingerprint)
    summary = pd.DataFrame(summaries, columns=[*original.columns, *SUMMARY_COLUMNS])
    detail_columns = ['cycle_id', 'start_index', 'end_index', 'automatic_state', 'new_automatic_state',
                      'trial_max_segments', 'trial_noise_multiplier', 'trial_min_segment_samples',
                      'trial_max_boundary_candidates']
    detail = pd.DataFrame(details) if details else pd.DataFrame(columns=detail_columns + event_columns)
    metadata = {
        'audit_version': AUDIT_VERSION, 'feature_version': MODEL_VERSION,
        'event_model_version': EVENT_MODEL_VERSION,
        'source_file': str(source_path.resolve()), 'source_sha256': source_hash,
        'expected_source_sha256': expected, 'independent_source_sha256_verified': True,
        'source_samples': source_samples, 'waveform_data_loaded': not selected.empty,
        'results_file': str(result_path.resolve()), 'results_sha256': result_hash,
        'model_config_file': str(config_path.resolve()), 'model_config_sha256': config_hash,
        'recorded_dynamic_config': recorded_config, 'baseline_event_config': asdict(base),
        'baseline_max_segments': BASELINE_BUDGET, 'trial_configs': [asdict(value) for value in trial_configs],
        'selection': ('all_new_effective_suggestions_including_retained' if include_retained
                      else 'new_effective_suggestions_not_previously_effective'),
        'include_retained': bool(include_retained), 'selected_cycle_ids': [int(value) for value in selected['cycle_id']],
        'selected_cycles': len(selected), 'original_result_cycles': len(original),
        'empty_scope': selected.empty, 'baseline_replay_matched_cycles': len(selected),
        'selected_raw_features_matched_cycles': len(selected),
        'supported_all_budgets_cycles': int(summary['stability_supported_all_budgets'].sum()) if len(summary) else 0,
        'original_state_columns_preserved': True, 'manual_label_columns_preserved': True,
        'batch_model_refitted': False, 'boundaries_redetected': False, 'calibration_applied': False,
        'labels_changed_by_audit': False, 'state_probability_computed': False,
        'experimental_accuracy_assessed': False, 'audit_validates_candidate_labels': False,
        'event_bounds_are_proven_physical_onsets': False,
        'larger_segment_budget_is_more_correct': False,
        'diagnostic_note': '仅比较固定观测尺度/最小段长/边界网格下的分段预算 4/6/8/10；稳定不代表真值，变化不代表错误；分段更细并不保证证据更正确；事件边界是模型提案，未验证为物理起止点；摘要全部原列保留。',
        'known_development_control': {
            'source_filename': 'gyb1-4.csv',
            'source_sha256': '6faa67c6c37dedd0d39f4ce039c0bff9bdcb84baef3b88feeb06b429a313f047',
            'cycle_id': 570,
            'applies_to_this_source': source_hash == '6faa67c6c37dedd0d39f4ce039c0bff9bdcb84baef3b88feeb06b429a313f047',
            'note': '该开发源的周期 570 已获人工有效确认；片段探查中预算 4/6/8 支持、10 不支持。不是其他实验同编号的真值，也不用于本工具分类。',
        },
    }
    metadata_text = json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    folder = Path(output_dir) / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    detail.to_csv(folder / '分段稳定性明细.csv', index=False, encoding='utf-8-sig', float_format='%.17g')
    summary.to_csv(folder / '分段稳定性摘要.csv', index=False, encoding='utf-8-sig', float_format='%.17g')
    (folder / '运行来源.json').write_text(metadata_text, encoding='utf-8')
    print('分段预算稳定性审计周期数:', len(selected))
    print('全部预算均支持的周期数:', metadata['supported_all_budgets_cycles'])
    print('预算 6 与原事件记录重演匹配周期数:', len(selected))
    print('未改变候选状态；支持次数不是状态概率或实验准确率。')
    print('独立结果目录:', folder.resolve())
    return summary, detail, folder


def main(argv=None):
    parser = argparse.ArgumentParser(description='只读审计局部事件分段预算 4/6/8/10 的敏感性，保持标签')
    parser.add_argument('raw_source', help='完整原始 Dewesoft CSV')
    parser.add_argument('--results', required=True, help='已有局部事件复核结果.csv')
    parser.add_argument('--model-config', required=True, help='完整记录原 v4 DynamicConfig 的 JSON')
    parser.add_argument('--expected-source-sha256', required=True, help='已知原始 CSV SHA-256')
    parser.add_argument('--include-retained', action='store_true', help='同时审计原已有效的候选；默认只审计新增有效候选')
    parser.add_argument('--output', default='../输出结果/分段稳定性审计')
    args = parser.parse_args(argv)
    try:
        return audit_event_stability(args.raw_source, results=args.results, model_config=args.model_config,
                                     expected_source_sha256=args.expected_source_sha256, output_dir=args.output,
                                     include_retained=args.include_retained)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()

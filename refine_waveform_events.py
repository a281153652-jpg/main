"""在原周期边界上复核局部事件；输出独立候选结果，不改变 v4 主流程。"""
import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import time
from uuid import uuid4

import numpy as np
import pandas as pd

from modules.data_loader import load_waveform_data
from modules.放电状态识别 import (
    DECISION_RULE_VERSION, MODEL_VERSION, STATES, classify_with_evidence, extract_features,
)
from reclassify_features import _load_config, _validate_frame


SAMPLE_COLUMNS = ('time', 'voltage', 'current', 'cycle_id', 'source_index')
MANIFEST_COLUMNS = (
    'source_filename', 'source_sha256', 'source_samples',
    'source_hash_verification_passed', 'feature_file_sha256', 'calibration_applied',
)


def _sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _fingerprint(value, name):
    if not isinstance(value, str) or re.fullmatch(r'[0-9a-fA-F]{64}', value) is None:
        raise ValueError(f'{name} 必须为 64 位十六进制 SHA-256')
    return value.lower()


def _snapshot(path):
    path = Path(path)
    snapshot = path.read_bytes()
    return snapshot, hashlib.sha256(snapshot).hexdigest()


def _unchanged(path, fingerprint):
    if _sha256(path) != fingerprint:
        raise ValueError(f'复核期间输入文件发生变化，未写出结果: {path}')


def _load_event_model(dynamic_config):
    # 延迟加载使输入校验可独立进行；不修改现有主流程的模型选择。
    from modules import 局部事件模型 as model
    config = model.LocalEventConfig(noise_multiplier=dynamic_config.noise_multiplier,
                                   min_segment_samples=dynamic_config.min_samples)
    return model.EVENT_MODEL_VERSION, model.fit_local_events, config


def _waveform_valid(frame, context):
    for name in ('time', 'voltage', 'current'):
        if name not in frame:
            raise ValueError(f'{context} 缺少 {name}')
        values = pd.to_numeric(frame[name], errors='raise')
        if not np.isfinite(values).all():
            raise ValueError(f'{context} 的 {name} 必须全部有限，不能删点或插值')
        frame[name] = values
    if len(frame) < 2 or not np.all(np.diff(frame['time']) > 0):
        raise ValueError(f'{context} 的时间必须严格递增且至少有两个点')


def _one_manifest_value(frame, name):
    if frame[name].isna().any() or frame[name].nunique(dropna=False) != 1:
        raise ValueError(f'复核样本的 {name} 必须在全部行中记录同一个非空值')
    return frame[name].iloc[0]


def _boolean(value, name):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value in ('True', 'False'):
        return value == 'True'
    raise ValueError(f'{name} 必须明确记录 True 或 False')


def _integer_column(frame, name, minimum):
    """拒绝布尔值和 int64 上界的浮点舍入，避免索引静默溢出。"""
    if frame[name].map(lambda value: isinstance(value, (bool, np.bool_))).any():
        raise ValueError(f'{name} 必须为不小于 {minimum} 的有限整数，不能是布尔值')
    values = pd.to_numeric(frame[name], errors='raise')
    if (not np.isfinite(values).all() or not values.ge(minimum).all()
            or not values.mod(1).eq(0).all()
            or not values.map(lambda value: int(value) <= np.iinfo(np.int64).max).all()):
        raise ValueError(f'{name} 必须为不小于 {minimum} 且不超出 int64 的有限整数')
    frame[name] = values.astype('int64')


def _review_blocks(samples, features, expected):
    missing = set(SAMPLE_COLUMNS + MANIFEST_COLUMNS) - set(samples)
    if missing:
        raise ValueError(f'复核样本缺少逐点数据或来源列: {sorted(missing)}')
    if samples.empty:
        raise ValueError('复核样本没有记录')
    manifest = {name: _one_manifest_value(samples, name) for name in MANIFEST_COLUMNS}
    manifest['source_sha256'] = _fingerprint(manifest['source_sha256'], 'source_sha256')
    manifest['feature_file_sha256'] = _fingerprint(manifest['feature_file_sha256'], 'feature_file_sha256')
    if manifest['source_sha256'] != expected:
        raise ValueError('复核样本声明的原始来源 SHA-256 与预期不一致')
    manifest['source_hash_verification_passed'] = _boolean(
        manifest['source_hash_verification_passed'], 'source_hash_verification_passed')
    if not manifest['source_hash_verification_passed']:
        raise ValueError('复核样本未声明在导出时通过原始来源哈希核对')
    manifest['calibration_applied'] = _boolean(manifest['calibration_applied'], 'calibration_applied')
    if manifest['calibration_applied']:
        raise ValueError('只接受未另行校准的原始复核样本')
    count = pd.DataFrame({'source_samples': [manifest['source_samples']]})
    _integer_column(count, 'source_samples', 2)
    manifest['source_samples'] = int(count.loc[0, 'source_samples'])
    if not isinstance(manifest['source_filename'], str) or not manifest['source_filename'].strip():
        raise ValueError('source_filename 必须为非空文件名')
    for name, minimum in (('cycle_id', 1), ('source_index', 0)):
        _integer_column(samples, name, minimum)
    if samples['source_index'].duplicated().any():
        raise ValueError('复核样本 source_index 重复，不能确认原始逐点映射')
    if not samples['source_index'].lt(manifest['source_samples']).all():
        raise ValueError('复核样本 source_index 超出声明的原始样本数')
    unknown = set(samples['cycle_id']) - set(features['cycle_id'])
    if unknown:
        raise ValueError(f'复核样本含结果表中没有的周期: {sorted(unknown)}')
    blocks, time_origins, time_tolerances = [], [], []
    for cycle_id, group in samples.groupby('cycle_id', sort=False):
        row = features.loc[features['cycle_id'].eq(cycle_id)].iloc[0]
        indices = np.arange(int(row['start_index']), int(row['end_index']), dtype=np.int64)
        if len(group) != int(row['sample_count']) or not np.array_equal(group['source_index'], indices):
            raise ValueError(f'周期 {cycle_id} 必须包含原左闭右开边界内按顺序排列的全部样本')
        block = group.loc[:, ['time', 'voltage', 'current']].copy().reset_index(drop=True)
        _waveform_valid(block, f'周期 {cycle_id}')
        # 原表无绝对起始时间；这里只核对各片段的时间与来源索引相容。
        sample_interval = 1 / float(row['sampling_rate_Hz'])
        if not np.isfinite(sample_interval) or sample_interval <= 0:
            raise ValueError(f'周期 {cycle_id} 的原采样率必须为有限正数')
        time_origins.append(float(block['time'].iloc[0]) - int(row['start_index']) * sample_interval)
        time_tolerances.append(.01 * sample_interval + np.finfo(float).eps
                               * max(1., float(block['time'].abs().max())) * 32)
        blocks.append((row, block))
    if len(time_origins) > 1 and np.ptp(time_origins) > 2 * max(time_tolerances):
        raise ValueError('复核片段的时间与 source_index/原采样率无法对应同一个时间起点')
    return blocks, manifest


def _match_features(recorded, recomputed):
    """同一周期的全部 v4 波形描述量应一致；旧判别规则标签可以不同。"""
    for name, actual in recomputed.items():
        if name not in recorded:
            raise ValueError(f'周期 {recorded["cycle_id"]} 缺少原始 v4 特征: {name}')
        previous = recorded[name]
        if isinstance(actual, (bool, np.bool_)):
            equal = _boolean(previous, name) == bool(actual)
        elif isinstance(actual, str):
            equal = previous == actual
        elif pd.isna(actual):
            equal = pd.isna(previous)
        else:
            try:
                equal = bool(np.isclose(float(previous), float(actual), rtol=1e-9, atol=1e-12))
            except (TypeError, ValueError):
                equal = False
        if not equal:
            raise ValueError(f'周期 {recorded["cycle_id"]} 的原始 v4 特征与逐点重提不一致: {name}')


def _new_decision(row, recomputed, base_state, event):
    old_dynamic = row['state']
    old_automatic = row.get('automatic_state',
                            'undetermined' if old_dynamic == 'abnormal_discharge' else old_dynamic)
    # 新事件证据只增加候选，已有开路/短路/有效标签保留；未知噪声尺度优先。
    if recomputed['data_quality'] != 'ok' or not recomputed.get('current_scale_reliable', False):
        return old_dynamic, 'undetermined', 'untrusted_recomputed_current_scale'
    eligible = (old_dynamic == 'abnormal_discharge'
                and old_automatic in ('undetermined', 'abnormal_discharge')
                and base_state == 'abnormal_discharge')
    if eligible and event['event_supported']:
        return 'effective_discharge', 'effective_discharge', 'local_event_joint_evidence_candidate'
    return old_dynamic, old_automatic, ('original_supported_state_preserved' if not eligible
                                        else event['event_reason'])


def refine_waveform_events(*, results, model_config, expected_source_sha256, output_dir,
                           raw_source=None, review_samples=None):
    if (raw_source is None) == (review_samples is None):
        raise ValueError('必须且只能提供 raw_source 或 --review-samples 其中一个')
    expected = _fingerprint(expected_source_sha256, '--expected-source-sha256')
    result_path, config_path = Path(results), Path(model_config)
    feature_snapshot, feature_hash = _snapshot(result_path)
    _, config_hash = _snapshot(config_path)
    config, recorded_config = _load_config(config_path)
    _unchanged(config_path, config_hash)
    original = pd.read_csv(BytesIO(feature_snapshot), encoding='utf-8-sig', float_precision='round_trip')
    validated = original.copy(deep=True)
    for name, minimum in (('cycle_id', 1), ('start_index', 0), ('end_index', 1), ('sample_count', 1)):
        if name in validated:
            _integer_column(validated, name, minimum)
    _validate_frame(validated, config)
    ordered = validated.sort_values('start_index')
    if (ordered['start_index'].to_numpy()[1:] < ordered['end_index'].to_numpy()[:-1]).any():
        raise ValueError('结果表的原周期边界存在重叠')
    if 'automatic_state' in validated and not validated['automatic_state'].isin((*STATES, 'undetermined')).all():
        raise ValueError('结果表含不支持的 automatic_state')
    if review_samples is not None:
        input_path = Path(review_samples)
        input_snapshot, input_hash = _snapshot(input_path)
        samples = pd.read_csv(BytesIO(input_snapshot), encoding='utf-8-sig', float_precision='round_trip',
                              dtype={name: 'string' for name in
                                     ('source_filename', 'source_sha256', 'feature_file_sha256')})
        blocks, manifest = _review_blocks(samples, validated, expected)
        source_samples = manifest['source_samples']
        if not validated['end_index'].le(source_samples).all():
            raise ValueError('结果表边界超出复核样本声明的原始总样本数')
        feature_hash_matches = manifest['feature_file_sha256'] == feature_hash
        source_identity = {
            'mode': 'review_samples', 'source_filename': manifest['source_filename'],
            'source_sha256': manifest['source_sha256'], 'source_samples': source_samples,
            'independent_source_sha256_verified': False,
            'uploaded_manifest_matches_expected_source_sha256': True,
            'exporter_declared_source_hash_verified': manifest['source_hash_verification_passed'],
            'manifest_feature_file_sha256': manifest['feature_file_sha256'],
            'manifest_feature_hash_matches_provided_results': feature_hash_matches,
            'identity_verification': 'selected_raw_features_and_bounds',
            'source_verification_note': '仅核对上传文件声明与预期；未独立读取或哈希完整原始 CSV',
            'absolute_time_origin_independently_verified': False,
            'time_check': 'per_cycle_time_descriptors_and_shared_time_index_origin',
        }
    else:
        input_path = Path(raw_source)
        input_hash = _sha256(input_path)
        if input_hash != expected:
            raise ValueError('原始 CSV SHA-256 与预期不一致；未读取波形或写出结果')
        data = load_waveform_data(input_path)
        _unchanged(input_path, input_hash)
        _waveform_valid(data, '原始数据')
        if not validated['end_index'].le(len(data)).all():
            raise ValueError('结果表周期边界超出原始 CSV 样本数')
        blocks = ((row, data.iloc[int(row['start_index']):int(row['end_index'])]
                   .copy().reset_index(drop=True)) for _, row in validated.iterrows())
        source_identity = {
            'mode': 'full_source', 'source_filename': input_path.name,
            'source_sha256': input_hash, 'source_samples': len(data),
            'independent_source_sha256_verified': True,
            'identity_verification': 'full_source_hash_selected_raw_features_and_bounds',
            'absolute_time_origin_independently_verified': True,
            'time_check': 'selected_original_source_samples',
        }
    version, event_fit, event_config = _load_event_model(config)
    output_rows = []
    last_progress = time.monotonic()
    total_to_process = len(blocks) if review_samples is not None else len(validated)
    for processed, (row, block) in enumerate(blocks, start=1):
        features = extract_features(block, config)
        _match_features(row, features)
        base_state, base_reason, base_short_score = classify_with_evidence(features, config)
        event = event_fit(block, features, config=event_config)
        if not isinstance(event, dict) or not {'event_supported', 'event_reason'} <= set(event):
            raise ValueError('局部事件模型必须返回 event_supported 和 event_reason 的平坦字典')
        event = event.copy()
        if any(not isinstance(name, str) or not name.startswith('event_') for name in event):
            raise ValueError('局部事件证据列必须以 event_ 开头，不能覆盖原始特征')
        event['event_supported'] = _boolean(event['event_supported'], 'event_supported')
        if not isinstance(event['event_reason'], str) or not event['event_reason']:
            raise ValueError('event_reason 必须为非空文本')
        for value in event.values():
            if isinstance(value, (dict, list, tuple, np.ndarray)):
                raise ValueError('局部事件证据必须为平坦标量')
            if isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)) and np.isinf(value):
                raise ValueError('局部事件证据不能含无限数值')
        new_state, new_automatic, reason = _new_decision(row, features, base_state, event)
        additions = {
            **event, 'event_model_version': version,
            'base_recomputed_state': base_state, 'base_recomputed_classification_reason': base_reason,
            'base_recomputed_short_score': base_short_score,
            'base_recomputed_decision_rule_version': DECISION_RULE_VERSION,
            'new_dynamic_state': new_state, 'new_automatic_state': new_automatic,
            'new_classification_reason': reason, 'new_automatic_label_validated': False,
            'local_event_state_changed': new_state != row['state'],
            'local_event_automatic_changed': new_automatic != row.get(
                'automatic_state', 'undetermined' if row['state'] == 'abnormal_discharge' else row['state']),
        }
        collisions = set(additions) & set(original.columns)
        if collisions:
            raise ValueError(f'输入已含本复核工具输出列，请使用原 v4 结果表: {sorted(collisions)}')
        preserved = original.loc[original['cycle_id'].eq(row['cycle_id'])].iloc[0].to_dict()
        output_rows.append({**preserved, **additions})
        if time.monotonic() - last_progress >= 20:
            print(f'局部事件复核进度: {processed}/{total_to_process}', flush=True)
            last_progress = time.monotonic()
    # 对来源、配置及原结果文件做最后核对；失败时不创建输出目录。
    for path, fingerprint in ((result_path, feature_hash), (config_path, config_hash), (input_path, input_hash)):
        _unchanged(path, fingerprint)
    output = pd.DataFrame(output_rows)
    folder = Path(output_dir) / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:8])
    metadata = {
        **source_identity, 'expected_source_sha256': expected, 'raw_input_file': str(input_path.resolve()),
        'raw_input_snapshot_sha256': input_hash, 'provided_results_file': str(result_path.resolve()),
        'provided_results_sha256': feature_hash, 'model_config_file': str(config_path.resolve()),
        'model_config_sha256': config_hash, 'recorded_dynamic_config': recorded_config,
        'feature_version': MODEL_VERSION, 'base_decision_rule_version': DECISION_RULE_VERSION,
        'event_model_version': version,
        'event_config': asdict(event_config) if is_dataclass(event_config) else None,
        'processed_cycle_ids': [int(value) for value in output['cycle_id']],
        'processed_cycles': len(output), 'original_result_cycles': len(original),
        'full_batch_processed': len(output) == len(original),
        'batch_model_refitted': False, 'boundaries_redetected': False, 'calibration_applied': False,
        'original_state_columns_preserved': True, 'manual_label_columns_preserved': True,
        'new_automatic_labels_validated': False, 'raw_features_match_original_results': True,
        'validation_note': '开发性候选复核，不代表实验准确率；子集不会重新拟合整批形态群组',
        'new_effective_candidates': int(output['local_event_state_changed'].sum()),
    }
    # JSON 严格序列化也应在目录创建前成功。
    metadata_text = json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    folder.mkdir(parents=True, exist_ok=False)
    output.to_csv(folder / '局部事件复核结果.csv', index=False, encoding='utf-8-sig', float_format='%.17g')
    (folder / '运行来源.json').write_text(metadata_text, encoding='utf-8')
    print('原周期边界复核数:', len(output))
    print('本次为整表复核。' if metadata['full_batch_processed'] else
          f'本次仅复核 {len(output)} 个已有周期；原整表共 {len(original)} 个周期。')
    print('局部事件新增有效放电候选:', metadata['new_effective_candidates'])
    print('本次 new_automatic_state 候选数量:')
    print(output['new_automatic_state'].value_counts().to_string())
    print('候选尚未独立验证；原始标签及整批群组保持。')
    print('独立结果目录:', folder.resolve())
    return output, folder


def main(argv=None):
    parser = argparse.ArgumentParser(description='按原周期边界复核局部事件，原 v4 输出及主模型保留')
    parser.add_argument('raw_source', nargs='?', help='完整原始 Dewesoft CSV；与 --review-samples 互斥')
    parser.add_argument('--review-samples', help='已导出的含来源信息的单个周期复核样本 CSV')
    parser.add_argument('--results', required=True, help='原完整 v4 放电状态或自学习结果 CSV')
    parser.add_argument('--model-config', required=True, help='完整记录 DynamicConfig 的识别模型参数.json')
    parser.add_argument('--expected-source-sha256', required=True, help='已知原始 CSV SHA-256')
    parser.add_argument('--output', default='../输出结果/局部事件复核')
    args = parser.parse_args(argv)
    try:
        return refine_waveform_events(
            results=args.results, model_config=args.model_config,
            expected_source_sha256=args.expected_source_sha256, output_dir=args.output,
            raw_source=args.raw_source, review_samples=args.review_samples)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()

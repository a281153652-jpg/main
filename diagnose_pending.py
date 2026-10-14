"""导出少量待定证据，保留现有标签；可选收集来源核对后的原始周期。"""
import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
from uuid import uuid4

import pandas as pd

from modules.data_loader import load_waveform_data
from modules.待定诊断 import diagnose_pending
from modules.周期复核导出 import export_review_cycles


def _positive_integer(value):
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError('必须为正整数') from error
    if number <= 0:
        raise argparse.ArgumentTypeError('必须为正整数')
    return number


def _sha256(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description='诊断 v4 待定周期，只收集证据，不更改任何状态')
    parser.add_argument('feature_file', help='v4 自学习状态结果.csv')
    parser.add_argument('--output', default='../输出结果/待定诊断_v4', help='每次建立唯一子目录，不覆盖旧结果')
    parser.add_argument('--examples', type=_positive_integer, default=12, help='代表周期预算，默认 12')
    parser.add_argument('--source', help='可选原始实验 CSV，仅用于逐点证据收集，不更改状态')
    parser.add_argument('--expected-source-sha256', help='使用 --source 时必需：原始 CSV 的已知 SHA-256')
    args = parser.parse_args(argv)
    feature_path = Path(args.feature_file)
    source_path = Path(args.source) if args.source else None
    if not feature_path.is_file():
        parser.error(f'找不到自学习结果: {feature_path}')
    if bool(source_path) != bool(args.expected_source_sha256):
        parser.error('--source 与 --expected-source-sha256 必须一起提供；两者都省略可仅诊断特征')
    expected = args.expected_source_sha256.lower() if args.expected_source_sha256 else None
    if expected and not re.fullmatch(r'[0-9a-f]{64}', expected):
        parser.error('--expected-source-sha256 必须为 64 位十六进制 SHA-256')
    source_hash = None
    if source_path:
        if not source_path.is_file():
            parser.error(f'找不到原始实验 CSV: {source_path}')
    # 同一字节快照用于哈希与解析，记录输入准确来源，不修改输入文件。
    feature_bytes = feature_path.read_bytes()
    results = pd.read_csv(BytesIO(feature_bytes), encoding='utf-8-sig', float_precision='round_trip')
    try:
        counts, summary, representatives, metadata = diagnose_pending(results, args.examples)
    except ValueError as error:
        parser.error(str(error))
    selected = representatives['cycle_id'].astype(int).tolist()
    data = None
    if source_path and selected:
        print('核对原始文件 SHA-256；通过后才读取波形或写入报告。', flush=True)
        source_hash = _sha256(source_path)
        if source_hash != expected:
            parser.error(f'原始文件 SHA-256 不匹配，未生成报告。\n实际: {source_hash}\n预期: {expected}')
        data = load_waveform_data(source_path)
        # 防止校验后文件被编辑，尚未写出任何诊断文件。
        if _sha256(source_path) != source_hash:
            parser.error('原始文件在读取期间发生变化，未生成报告，请重试')
    metadata.update({
        'input_feature_file': feature_path.name,
        'input_feature_file_sha256': hashlib.sha256(feature_bytes).hexdigest(),
        'source_filename': source_path.name if source_path else None,
        'source_sha256': source_hash, 'expected_source_sha256': expected,
        'source_hash_verification_passed': source_hash == expected if source_hash is not None else None,
        'source_samples': len(data) if data is not None else None,
        'raw_samples_exported': False, 'calibration_applied': False,
        'raw_export_skip_reason': ('no_pending_cycles' if not selected
                                   else 'no_source_provided' if not source_path else None),
    })
    batch = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:8]
    folder = Path(args.output) / batch
    folder.mkdir(parents=True, exist_ok=False)
    if data is not None:
        try:
            path, samples = export_review_cycles(data, results, folder, selected)
        except (ValueError, FileExistsError) as error:
            parser.error(str(error))
        source_metadata_path = folder / '周期复核来源.json'
        source_metadata = json.loads(source_metadata_path.read_text(encoding='utf-8'))
        source_metadata.update({'source_filename': source_path.name, 'source_sha256': source_hash,
                                'source_samples': len(data), 'source_hash_verification_passed': True,
                                'expected_source_sha256': expected})
        source_metadata_path.write_text(json.dumps(source_metadata, ensure_ascii=False, indent=2,
                                                  allow_nan=False) + '\n', encoding='utf-8')
        metadata['raw_samples_exported'] = True
        metadata['raw_sample_count'] = len(samples)
        print('已收集原始逐点证据:', path)
    counts.to_csv(folder / '待定原因统计.csv', index=False, encoding='utf-8-sig')
    summary.to_csv(folder / '待定特征摘要.csv', index=False, encoding='utf-8-sig')
    representatives.to_csv(folder / '诊断代表周期.csv', index=False, encoding='utf-8-sig')
    (folder / '诊断说明.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2,
                                                   allow_nan=False) + '\n', encoding='utf-8')
    print('\n待定原因总表（缺失原因也保留）:')
    print(counts.to_string(index=False) if len(counts) else '没有待定周期，无需读取或导出原始波形。')
    print('\n代表周期:', selected)
    print('原因组覆盖:', metadata['covered_reason_groups'], '/', metadata['pending_reason_groups'])
    if metadata['uncovered_reason_groups']:
        print('预算未覆盖的原因组已列入诊断说明.json；可增加 --examples。')
    print('本步骤只收集证据，没有更改状态；SNR 与群组归属不是加工状态概率。')
    print('输出目录:', folder)
    print('请回传小文件: 待定特征摘要.csv、诊断代表周期.csv、诊断说明.json')
    if metadata['raw_samples_exported']:
        print('逐点复核时另回传: 周期复核样本.csv、周期复核来源.json')
    return folder


if __name__ == '__main__':
    main()

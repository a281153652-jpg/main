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
from modules.周期复核导出 import export_review_cycles


def source_sha256(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description='导出少量完整周期供逐点复核，保留原始测量值')
    parser.add_argument('input_file', nargs='?', default='../输入数据/gyb1-4.csv')
    parser.add_argument('--results', default='../输出结果/动态特征复核/放电状态结果.csv')
    parser.add_argument('--ids', default='29,103,570,605', help='逗号分隔的周期编号')
    parser.add_argument('--output', default='../输出结果/周期复核包')
    parser.add_argument('--expected-source-sha256', help='核对原始 CSV 的已知 SHA-256')
    parser.add_argument('--single-file', action='store_true',
                        help='在新导出的样本 CSV 内附来源信息，复核时只需上传这一个 CSV')
    args = parser.parse_args(argv)
    expected = args.expected_source_sha256
    if expected is not None:
        if re.fullmatch(r'[0-9a-fA-F]{64}', expected) is None:
            parser.error('--expected-source-sha256 必须为 64 位十六进制 SHA-256')
        expected = expected.lower()
    try:
        ids = [int(value.strip()) for value in args.ids.split(',')]
    except ValueError:
        parser.error('--ids 必须是逗号分隔的整数')
    if any(value <= 0 for value in ids):
        parser.error('周期编号必须为正整数')
    if not Path(args.input_file).is_file() or not Path(args.results).is_file():
        parser.error('找不到输入 CSV 或原有结果表，请核对路径')
    source_hash = source_sha256(args.input_file)
    if expected is not None and source_hash != expected:
        raise ValueError('原始 CSV 的 SHA-256 与预期不一致；未读取波形或写出复核文件')
    feature_snapshot = Path(args.results).read_bytes()
    feature_hash = hashlib.sha256(feature_snapshot).hexdigest()
    results = pd.read_csv(BytesIO(feature_snapshot), float_precision='round_trip')
    data = load_waveform_data(args.input_file)
    if source_sha256(args.input_file) != source_hash:
        raise ValueError('读取期间原始 CSV 发生变化；未写出复核文件，请重试')
    batch = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:6]
    folder = Path(args.output) / batch
    path, samples = export_review_cycles(data, results, folder, ids)
    provenance = {'source_filename': Path(args.input_file).name,
                  'source_sha256': source_hash, 'source_samples': len(data),
                  'source_hash_verification_passed': True if expected is not None else None,
                  'feature_file_sha256': feature_hash, 'calibration_applied': False}
    if args.single_file:
        upload_samples = samples.copy(deep=True)
        for name, value in provenance.items():
            upload_samples[name] = value
        upload_samples.to_csv(path, index=False, encoding='utf-8-sig', float_format='%.17g')
    metadata_path = folder / '周期复核来源.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    metadata.update(provenance)
    metadata['expected_source_sha256'] = expected
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print('已导出周期:', ids)
    print('逐点样本数:', len(samples))
    print('文件大小:', path.stat().st_size, '字节')
    print('来源 SHA-256:', source_hash)
    print('来源哈希核对:', '通过' if expected is not None else '未提供预期哈希，仅记录指纹')
    if args.single_file:
        print('本次只需上传周期复核样本.csv:', path)
    else:
        print('请上传此小 CSV 和同目录的来源 JSON:', path)
    return path


if __name__ == '__main__':
    main()

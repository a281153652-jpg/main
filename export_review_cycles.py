import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import pandas as pd

from modules.data_loader import load_waveform_data
from modules.周期复核导出 import export_review_cycles


def main(argv=None):
    parser = argparse.ArgumentParser(description='导出少量完整周期供逐点复核，保留原始测量值')
    parser.add_argument('input_file', nargs='?', default='../输入数据/gyb1-4.csv')
    parser.add_argument('--results', default='../输出结果/动态特征复核/放电状态结果.csv')
    parser.add_argument('--ids', default='29,103,570,605', help='逗号分隔的周期编号')
    parser.add_argument('--output', default='../输出结果/周期复核包')
    args = parser.parse_args(argv)
    try:
        ids = [int(value.strip()) for value in args.ids.split(',')]
    except ValueError:
        parser.error('--ids 必须是逗号分隔的整数')
    if any(value <= 0 for value in ids):
        parser.error('周期编号必须为正整数')
    if not Path(args.input_file).is_file() or not Path(args.results).is_file():
        parser.error('找不到输入 CSV 或原有结果表，请核对路径')
    data = load_waveform_data(args.input_file)
    results = pd.read_csv(args.results)
    batch = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:6]
    folder = Path(args.output) / batch
    path, samples = export_review_cycles(data, results, folder, ids)
    with open(args.input_file, 'rb') as source:
        source_hash = hashlib.file_digest(source, 'sha256').hexdigest()
    metadata_path = folder / '周期复核来源.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    metadata.update({'source_filename': Path(args.input_file).name, 'source_sha256': source_hash,
                     'source_samples': len(data)})
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print('已导出周期:', ids)
    print('逐点样本数:', len(samples))
    print('文件大小:', path.stat().st_size, '字节')
    print('请上传此小 CSV 和同目录的来源 JSON:', path)
    return path


if __name__ == '__main__':
    main()

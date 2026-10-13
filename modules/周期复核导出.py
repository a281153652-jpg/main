"""导出少量完整原始周期，保留逐点数据和原始位置，不重新检测边界。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def export_review_cycles(data, results, output_dir, cycle_ids):
    required = {'cycle_id', 'start_index', 'end_index'}
    if not required.issubset(results):
        raise ValueError('结果表缺少周期编号或边界列')
    if results['cycle_id'].duplicated().any():
        raise ValueError('周期编号重复，不能确定原始边界')
    ids = list(dict.fromkeys(cycle_ids))
    if not ids:
        raise ValueError('至少指定一个周期')
    missing = sorted(set(ids) - set(results['cycle_id']))
    if missing:
        raise ValueError(f'结果表中找不到周期: {missing}')
    blocks, mappings = [], []
    for cycle_id in ids:
        row = results.loc[results['cycle_id'] == cycle_id].iloc[0]
        start, end = row['start_index'], row['end_index']
        if (not np.isfinite(start) or not np.isfinite(end)
                or int(start) != start or int(end) != end or not 0 <= start < end <= len(data)):
            raise ValueError(f'周期 {cycle_id} 的原始边界不合法')
        start, end = int(start), int(end)
        if 'sample_count' in row and row['sample_count'] != end - start:
            raise ValueError(f'周期 {cycle_id} 的边界长度与结果样本数不一致')
        block = data.iloc[start:end][['time', 'voltage', 'current']].copy()
        block['cycle_id'] = int(cycle_id)
        block['source_index'] = np.arange(start, end)
        blocks.append(block)
        mappings.append({'cycle_id': int(cycle_id), 'source_start_index': start,
                         'source_end_index': end, 'sample_count': end - start,
                         'result_model_version': row.get('model_version', 'unknown')})
    samples = pd.concat(blocks, ignore_index=True)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / '周期复核样本.csv'
    json_path = output / '周期复核来源.json'
    if csv_path.exists() or json_path.exists():
        raise FileExistsError('复核文件已存在，请指定新的输出目录')
    samples.to_csv(csv_path, index=False, encoding='utf-8-sig', float_format='%.17g')
    json_path.write_text(json.dumps({'cycles': mappings, 'calibration_applied': False,
                                    'source_indices_end_exclusive': True},
                                   ensure_ascii=False, indent=2), encoding='utf-8')
    return csv_path, samples

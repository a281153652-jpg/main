"""按新模型的联合证据排序；候选不等同于已确认短路。"""
from pathlib import Path

import numpy as np
import pandas as pd
from modules.放电状态识别 import MODEL_VERSION


def calculate_short_score(row):
    if 'short_score' not in row or not np.isfinite(row['short_score']):
        raise ValueError('结果不含有效的新模型 short_score，请重新运行放电状态识别')
    return float(row['short_score'])


def short_candidate_analysis(result_file, output_dir):
    df = pd.read_csv(result_file)
    required = {'short_score', 'data_quality', 'state', 'cycle_id'}
    if not required.issubset(df):
        raise ValueError('候选排序需要新模型输出，请重新运行放电状态识别')
    if not df.empty and ('model_version' not in df or not df['model_version'].eq(MODEL_VERSION).all()):
        raise ValueError('候选表模型版本过旧，必须从原始波形重新运行识别')
    # 损坏数据和开路不进入短路候选，避免凑满候选数量。
    candidate = df.loc[(df['data_quality'] == 'ok') & (df['state'] != 'open_circuit')
                      & (df['short_score'] > 0)].copy()
    if not candidate.empty:
        candidate['short_score'] = candidate.apply(calculate_short_score, axis=1)
    candidate = candidate.sort_values(['short_score', 'cycle_id'], ascending=[False, True])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    candidate.to_csv(output / '短路候选排序.csv', index=False, encoding='utf-8-sig')
    columns = ['cycle_id', 'state', 'short_score', 'oscillation_index',
               'post_sample_count', 'plateau_supported', 'post_current_std',
               'voltage_collapse_ratio', 'classification_reason']
    print('\n短路候选（评分不是概率，需人工复核）:')
    print(candidate[[column for column in columns if column in candidate]].head(20))
    return candidate.head(20)

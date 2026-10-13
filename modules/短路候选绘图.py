from pathlib import Path

import pandas as pd

from modules.data_loader import load_waveform_data
from modules.周期检测 import detect_cycles
from modules.状态验证绘图 import plot_cycle


def plot_short_candidates(input_file, score_file, output_dir, number=10, *, data=None, cycles=None):
    """保持旧调用方式，也允许复用内存中的数据和周期，不重复读取千万点 CSV。"""
    if number < 1:
        raise ValueError('候选图片数量必须为正整数')
    candidate = pd.read_csv(score_file).head(number)
    if candidate.empty:
        print('没有可绘制的短路候选')
        return []
    if data is None:
        data = load_waveform_data(input_file)
    if not {'start_index', 'end_index'}.issubset(candidate):
        if cycles is None:
            cycles = detect_cycles(data, output_dir)
        candidate = candidate.merge(cycles[['cycle_id', 'start_index', 'end_index']],
                                    on='cycle_id', how='left', validate='one_to_one')
        if candidate[['start_index', 'end_index']].isna().any().any():
            raise ValueError('候选周期无法匹配周期边界，请重新生成候选表')
    paths = []
    for _, row in candidate.iterrows():
        path = Path(output_dir) / f"cycle_{int(row['cycle_id'])}.png"
        title = f"Short candidate cycle {int(row['cycle_id'])} | evidence score={row['short_score']:.2f}"
        plot_cycle(data, row, path, title=title)
        paths.append(path)
    return paths

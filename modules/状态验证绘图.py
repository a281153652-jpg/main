"""四类波形及复核清单，使用分类时的周期边界，保留有符号电流。"""
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modules.放电状态识别 import STATES


def plot_cycle(data, row, save_path, title=None):
    cycle = data.iloc[int(row['start_index']):int(row['end_index'])]
    time = cycle['time'].to_numpy(dtype=float)
    relative_time = (time - time[0]) * 1e6
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    try:
        axes[0].plot(relative_time, cycle['voltage'], color='tab:blue', label='Voltage')
        axes[1].plot(relative_time, cycle['current'], color='tab:orange', label='Signed current')
        axes[0].set_ylabel('Voltage (V)')
        axes[1].set_ylabel('Current (A)')
        axes[1].set_xlabel('Time from cycle start (us)')
        for axis in axes:
            axis.legend(loc='best')
            axis.grid(alpha=.2)
        fig.suptitle(title or f"Cycle {row['cycle_id']} | {row['state']}\n"
                             f"{row.get('model_version', 'unknown version')}\n{row['classification_reason']}")
        fig.tight_layout()
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)


def plot_state_examples(data, results, output_dir, number=10):
    if number < 1:
        raise ValueError('每类样本数量必须为正整数')
    output = Path(output_dir)
    batch = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:6]
    folder = output / '状态验证' / batch
    manifest, summary = [], []
    for state in STATES:
        group = results.loc[results['state'] == state]
        order = [column for column in ('oscillation_index', 'robust_I_pp', 'cycle_id') if column in group]
        group = group.sort_values(order)
        count = min(number, len(group))
        # 跨特征分布抽样，避免只挑最像预期的波形。
        selected = group.iloc[np.linspace(0, len(group) - 1, count, dtype=int)] if count else group
        for _, row in selected.iterrows():
            path = folder / state / f"cycle_{int(row['cycle_id'])}.png"
            plot_cycle(data, row, path)
            manifest.append({
                'cycle_id': row['cycle_id'], 'suggested_state': state,
                'classification_reason': row['classification_reason'],
                'data_quality': row['data_quality'], 'image_path': str(path.relative_to(output)),
                'automatic_state': row.get('automatic_state', state),
                'review_priority': row.get('review_priority', 'high'),
                'morphology_cluster': row.get('morphology_cluster', -1),
                'reviewed_state': '', 'review_notes': '',
            })
        summary.append({'state': state, 'available_cycles': len(group), 'plotted_cycles': count,
                        'requested_cycles': number, 'status': 'enough' if count == number else 'insufficient'})
    output.mkdir(parents=True, exist_ok=True)
    checklist = pd.DataFrame(manifest, columns=['cycle_id', 'suggested_state', 'classification_reason',
                                               'data_quality', 'image_path', 'automatic_state', 'review_priority',
                                               'morphology_cluster', 'reviewed_state', 'review_notes'])
    # 新批次清单保存在图片目录中，保留已有人工复核记录。
    folder.mkdir(parents=True, exist_ok=True)
    checklist.to_csv(folder / '人工复核清单.csv', index=False, encoding='utf-8-sig')
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(folder / '状态样本统计.csv', index=False, encoding='utf-8-sig')
    print('\n本次状态样本输出:')
    print(summary_df.to_string(index=False))
    print('人工复核目录:', folder)
    return checklist

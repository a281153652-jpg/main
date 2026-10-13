import argparse
import hashlib
import json
import os
from pathlib import Path
import pandas as pd

# 云环境中使用可写缓存；图像保存不依赖桌面窗口。
os.environ.setdefault('MPLCONFIGDIR', '/tmp/discharge-matplotlib')
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/discharge-cache')

from modules.data_loader import load_waveform_data
from modules.周期检测 import detect_cycles
from modules.放电状态识别 import DynamicConfig, recognize_discharge_states
from modules.状态特征分析 import analyze_features
from modules.短路候选分析 import short_candidate_analysis
from modules.短路候选绘图 import plot_short_candidates
from modules.状态验证绘图 import plot_state_examples
from modules.批次自学习 import learn_batch_states


def main(argv=None):
    parser = argparse.ArgumentParser(description='高低压复合放电波形分析（动态识别与整批特征自学习）')
    parser.add_argument('input_file', nargs='?', default='./输入数据/gyb1-4.csv', help='Dewesoft CSV 文件')
    parser.add_argument('--output', default='./输出结果', help='结果目录')
    parser.add_argument('--examples', type=int, default=10, help='每类典型波形数量，默认 10')
    parser.add_argument('--current-offset', type=float, default=0.0,
                        help='已测定的零电流偏置(A)，默认不校正；不是状态分类阈值')
    parser.add_argument('--cycles', help='复用同一 CSV 先前自动检测的周期表，以固定边界比较模型')
    parser.add_argument('--learning', choices=('batch', 'off'), default='batch',
                        help='默认自动学习本批次形态分布；off 仅运行动态筛查')
    parser.add_argument('--expected-source-sha256', help='核对原始 CSV 的已知 SHA-256，避免复用错误文件的周期表')
    args = parser.parse_args(argv)
    if not Path(args.input_file).is_file():
        parser.error(f'找不到输入数据: {args.input_file}')
    if args.examples < 1:
        parser.error('--examples 必须为正整数')
    if args.expected_source_sha256 is not None:
        expected_hash = args.expected_source_sha256.lower()
        if len(expected_hash) != 64 or any(c not in '0123456789abcdef' for c in expected_hash):
            parser.error('--expected-source-sha256 必须为 64 位十六进制字符串')
    with open(args.input_file, 'rb') as source:
        source_hash = hashlib.file_digest(source, 'sha256').hexdigest()
    if args.expected_source_sha256 and source_hash != args.expected_source_sha256.lower():
        parser.error('原始 CSV SHA-256 与预期不一致，未执行识别或覆盖结果')
    output = Path(args.output)
    print('高低压复合放电分析系统\n分析文件:', args.input_file)
    data = load_waveform_data(args.input_file)
    if args.cycles:
        if not Path(args.cycles).is_file():
            parser.error(f'找不到自动周期表: {args.cycles}')
        cycles = pd.read_csv(args.cycles)
        print('复用自动周期边界:', args.cycles)
    else:
        cycles = detect_cycles(data, str(output))
    if cycles.empty:
        raise ValueError('未检测到完整周期，未执行状态识别')
    config = DynamicConfig(current_zero_offset=args.current_offset)
    result = recognize_discharge_states(data, cycles, output, config)
    if args.learning == 'batch':
        feature_file = output / '放电状态结果.csv'
        result = learn_batch_states(result, output, source_info={
            'feature_file_name': feature_file.name,
            'feature_file_sha256': hashlib.sha256(feature_file.read_bytes()).hexdigest(),
        })
    if args.cycles:
        cycles.to_csv(output / '周期信息.csv', index=False, encoding='utf-8-sig')
    (output / '运行来源.json').write_text(json.dumps({
        'input_filename': Path(args.input_file).name, 'source_sha256': source_hash,
        'input_samples': len(data), 'first_time_s': float(data['time'].iloc[0]),
        'last_time_s': float(data['time'].iloc[-1]),
        'cycles_reused_from': args.cycles, 'experimental_labels_validated': False,
        'expected_source_sha256': args.expected_source_sha256,
        'source_hash_matched_expected': bool(args.expected_source_sha256),
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    result_file = output / '放电状态结果.csv'
    analyze_features(result_file, output)
    short_candidate_analysis(result_file, output)
    plot_state_examples(data, result, output, args.examples)
    plot_short_candidates(args.input_file, output / '短路候选排序.csv', output / '短路候选图',
                          args.examples, data=data, cycles=cycles)
    print('\n分析完成：状态已自动输出，可抽查典型周期，优先复核证据不足的周期。')
    return result


if __name__ == '__main__':
    main()

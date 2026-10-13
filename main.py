import argparse
import os
from pathlib import Path

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


def main(argv=None):
    parser = argparse.ArgumentParser(description='高低压复合放电波形分析（自动筛查，需人工复核）')
    parser.add_argument('input_file', nargs='?', default='./输入数据/gyb1-4.csv', help='Dewesoft CSV 文件')
    parser.add_argument('--output', default='./输出结果', help='结果目录')
    parser.add_argument('--examples', type=int, default=10, help='每类典型波形数量，默认 10')
    parser.add_argument('--current-offset', type=float, default=0.0,
                        help='已测定的零电流偏置(A)，默认不校正；不是状态分类阈值')
    args = parser.parse_args(argv)
    if not Path(args.input_file).is_file():
        parser.error(f'找不到输入数据: {args.input_file}')
    if args.examples < 1:
        parser.error('--examples 必须为正整数')
    output = Path(args.output)
    print('高低压复合放电分析系统\n分析文件:', args.input_file)
    data = load_waveform_data(args.input_file)
    cycles = detect_cycles(data, str(output))
    if cycles.empty:
        raise ValueError('未检测到完整周期，未执行状态识别')
    config = DynamicConfig(current_zero_offset=args.current_offset)
    result = recognize_discharge_states(data, cycles, output, config)
    result_file = output / '放电状态结果.csv'
    analyze_features(result_file, output)
    short_candidate_analysis(result_file, output)
    plot_state_examples(data, result, output, args.examples)
    plot_short_candidates(args.input_file, output / '短路候选排序.csv', output / '短路候选图',
                          args.examples, data=data, cycles=cycles)
    print('\n分析完成：以上为自动状态建议，请使用本批次人工复核清单确认。')
    return result


if __name__ == '__main__':
    main()

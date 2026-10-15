"""新实验数据的独立预检入口；记录数据质量，不生成加工状态标签。"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4


def main(argv=None):
    parser = argparse.ArgumentParser(description='分块检查新 Dewesoft CSV 的通道、时间和原始信号质量')
    parser.add_argument('input_file', help='待检查的新 CSV')
    parser.add_argument('--output', default='../输出结果/数据质量检查', help='独立报告根目录')
    parser.add_argument('--time-column', help='仅在自动识别有歧义时指定时间列的完整名称')
    parser.add_argument('--voltage-column', help='仅在自动识别有歧义时指定电压列的完整名称')
    parser.add_argument('--current-column', help='仅在自动识别有歧义时指定电流列的完整名称')
    args = parser.parse_args(argv)
    source = Path(args.input_file)
    if not source.is_file():
        parser.error(f'找不到输入 CSV，请核对文件名、后缀和路径: {source}')
    print('新实验数据质量检查\n输入文件:', source.resolve(), flush=True)
    print('加载数据检查依赖…', flush=True)
    from modules.数据质量检查 import inspect_csv

    def progress(stage, rows):
        print(f'数据检查进度: {stage}，已处理 {rows} 行。', flush=True)

    try:
        report = inspect_csv(source, time_column=args.time_column,
                             voltage_column=args.voltage_column,
                             current_column=args.current_column, progress=progress)
        # 严格序列化在创建报告目录之前完成，拒绝 NaN/Infinity 冒充正常统计量。
        serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    folder = Path(args.output) / (source.stem + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
                                 + '_' + uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    path = folder / '数据质量报告.json'
    path.write_text(serialized, encoding='utf-8')
    print('数据点数量:', report['source_samples'])
    for role, label in (('time', '时间'), ('voltage', '电压'), ('current', '电流')):
        print(f'{label}列:', report['channels'][role]['column'])
    print('全量数值与时间检查:', '通过' if report['numeric_time_usable'] else '未通过，详见报告')
    print('该检查不代表完成仪器标定或放电状态验证。')
    print('数据质量报告:', path.resolve(), flush=True)
    return report, folder


if __name__ == '__main__':
    try:
        report, _ = main()
    except KeyboardInterrupt:
        print('数据检查已中断，可重新运行；原始数据保持。', flush=True)
        raise SystemExit(130) from None
    raise SystemExit(0 if report['numeric_time_usable'] else 2)

"""已有 v3 特征表可直接自学习，无需重新读取百万点波形。"""
import argparse
import hashlib
from pathlib import Path
import pandas as pd
from modules.批次自学习 import learn_batch_states


def main(argv=None):
    parser = argparse.ArgumentParser(description='从 v3 动态特征学习本批次的波形形态分布')
    parser.add_argument('feature_file', help='当前 v3 放电状态结果.csv')
    parser.add_argument('--output', default='./输出结果/批次自学习', help='独立学习结果目录')
    args = parser.parse_args(argv)
    path = Path(args.feature_file)
    if not path.is_file():
        parser.error(f'找不到特征表: {path}')
    output = Path(args.output)
    if (output / '自学习状态结果.csv').resolve() == path.resolve():
        parser.error('输入特征表不能是将被覆盖的输出文件')
    return learn_batch_states(pd.read_csv(path, float_precision='round_trip'), output,
                             source_info={'feature_file_name': path.name,
                                          'feature_file_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})


if __name__ == '__main__':
    main()

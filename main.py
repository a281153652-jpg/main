import os

from modules.data_loader import load_waveform_data
from modules.周期检测 import detect_cycles
from modules.放电状态识别 import recognize_discharge_states
from modules.状态特征分析 import analyze_features
from modules.短路候选分析 import short_candidate_analysis
from modules.短路候选绘图 import plot_short_candidates



def main():

    print("=" * 70)
    print("高低压复合放电分析系统")
    print("=" * 70)


    # =========================
    # 输入输出路径
    # =========================

    input_file = (
        "./输入数据/gyb1-4.csv"
    )


    output_dir = (
        "./输出结果"
    )


    print(
        "分析文件:",
        input_file
    )


    # =========================
    # 数据读取
    # =========================

    print("=" * 70)
    print("读取波形数据")
    print("=" * 70)


    data = load_waveform_data(
        input_file
    )


    print(
        "读取完成"
    )


    # =========================
    # 自动周期检测
    # =========================

    print("=" * 70)
    print("自动周期检测")
    print("=" * 70)


    cycles = detect_cycles(
        data,
        output_dir
    )


    print(
        "周期数量:",
        len(cycles)
    )


    # =========================
    # 单周期放电状态识别
    # =========================

    print("=" * 70)
    print("单周期放电状态识别")
    print("=" * 70)


    discharge_result = recognize_discharge_states(
        data,
        cycles,
        output_dir
    )
    analyze_features(
    "./输出结果/放电状态结果.csv",
    output_dir
    )
    short_candidate_analysis(
    "./输出结果/放电状态结果.csv",
    output_dir
    )
    plot_short_candidates(

    "./输入数据/gyb1-4.csv",

    "./输出结果/短路候选排序.csv",

    "./输出结果/短路候选图",

    10
    )


    print("=" * 70)
    print("全部分析完成")
    print("=" * 70)



if __name__ == "__main__":

    main()
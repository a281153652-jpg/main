import pandas as pd
import numpy as np


def load_waveform_data(filepath):

    print("=" * 70)
    print("读取波形数据")
    print("=" * 70)


    # 读取CSV
    df = pd.read_csv(filepath)


    print("数据点数量:")
    print(len(df))


    columns = df.columns.tolist()


    voltage_col = None
    current_col = None


    # 自动寻找电压、电流通道
    for col in columns:

        name = str(col).lower()


        if voltage_col is None:

            if (
                "voltage" in name
                or "电压" in name
                or "(v)" in name
                or "v (" in name
            ):
                voltage_col = col


        if current_col is None:

            if (
                "current" in name
                or "电流" in name
                or "(a)" in name
                or "a (" in name
            ):
                current_col = col



    if voltage_col is None:
        raise ValueError(
            "未自动识别到电压通道"
        )


    if current_col is None:
        raise ValueError(
            "未自动识别到电流通道"
        )


    print("\n识别通道:")
    print("-------------------------")

    print(
        "电压:",
        voltage_col
    )

    print(
        "电流:",
        current_col
    )


    # 建立标准数据格式

    data = pd.DataFrame()


    # 第一列默认时间
    data["time"] = df.iloc[:,0]


    data["voltage"] = (
        df[voltage_col]
        .astype(float)
    )


    data["current"] = (
        df[current_col]
        .astype(float)
    )


    # 计算采样频率

    dt = np.mean(
        np.diff(
            data["time"]
        )
    )


    fs = 1 / dt


    print("-------------------------")

    print(
        "采样频率:",
        round(fs,2),
        "Hz"
    )


    print(
        "读取完成"
    )


    return data
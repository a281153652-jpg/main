import os
import numpy as np
import pandas as pd

from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt



def lowpass_filter(signal, fs):

    """
    提取低频电压趋势
    """

    cutoff = min(
        fs / 100,
        fs * 0.02
    )

    b, a = butter(
        3,
        cutoff/(fs/2),
        btype="low"
    )

    return filtfilt(
        b,
        a,
        signal
    )



def fast_autocorrelation(signal):

    """
    FFT快速自相关
    """

    signal = (
        signal -
        np.mean(signal)
    )


    n = len(signal)


    fft = np.fft.fft(
        signal,
        n=2*n
    )


    acf = np.fft.ifft(
        fft*np.conj(fft)
    ).real


    return acf[:n]



def estimate_period(signal, fs):

    """
    单段周期估计
    """

    acf = fast_autocorrelation(
        signal
    )


    min_lag = int(
        fs*0.0001
    )

    max_lag = int(
        fs*0.01
    )


    search = acf[
        min_lag:max_lag
    ]


    peaks,_ = find_peaks(
        search
    )


    if len(peaks)==0:

        return None


    best = peaks[
        np.argmax(
            search[peaks]
        )
    ]


    period = (
        best+
        min_lag
    )


    return period



def multi_region_period(
        voltage,
        fs
):

    """
    前20% 中20% 后20%
    多区域周期估计
    """


    print("="*50)
    print("多区域周期估计")
    print("="*50)


    n=len(voltage)


    regions=[

        voltage[
            0:int(0.2*n)
        ],

        voltage[
            int(0.4*n):
            int(0.6*n)
        ],

        voltage[
            int(0.8*n):
        ]

    ]


    periods=[]


    for i,seg in enumerate(regions):


        print(
            "分析区域:",
            i+1
        )


        seg_low=lowpass_filter(
            seg,
            fs
        )


        p=estimate_period(
            seg_low,
            fs
        )


        print(
            "周期点数:",
            p
        )


        if p is not None:

            periods.append(p)



    if len(periods)==0:

        raise Exception(
            "周期估计失败"
        )


    periods=np.array(
        periods
    )


    print(
        "三个区域周期:",
        periods
    )


    mean=np.mean(
        periods
    )


    std=np.std(
        periods
    )


    cv=std/mean


    print(
        "周期平均:",
        mean,
        "点"
    )


    print(
        "周期波动:",
        cv*100,
        "%"
    )


    final_period=int(
        np.median(periods)
    )


    print(
        "最终周期:",
        final_period,
        "点"
    )


    return final_period



def locate_boundaries(
        voltage,
        period_points
):

    print(
        "周期边界定位..."
    )


    # 电压变化率

    dv = np.abs(
        np.diff(voltage)
    )


    # 第一步：
    # 找第一个周期起点

    search_length = int(
        period_points
    )


    first_region = dv[
        :search_length
    ]


    first_boundary = np.argmax(
        first_region
    )


    boundaries = [
        first_boundary
    ]


    print(
        "初始边界:",
        first_boundary
    )


    # 第二步：
    # 根据周期长度递推

    total_length = len(voltage)


    current = first_boundary


    while True:


        expected = (
            current
            +
            period_points
        )


        if expected >= total_length:

            break


        # 在预测位置附近搜索

        search_range = int(
            period_points*0.2
        )


        start = max(
            0,
            expected-search_range
        )


        end = min(
            len(dv),
            expected+search_range
        )


        local = dv[
            start:end
        ]


        if len(local)==0:

            break


        offset = np.argmax(
            local
        )


        real_position = (
            start
            +
            offset
        )


        boundaries.append(
            real_position
        )


        current = real_position



    boundaries = np.array(
        boundaries
    )


    return boundaries


    print(
        "周期边界定位..."
    )


    dv=np.abs(
        np.diff(voltage)
    )


    peaks,_=find_peaks(
        dv,
        distance=int(
            period_points*0.7
        )
    )


    boundaries=[]


    for p in peaks:


        if len(boundaries)==0:

            boundaries.append(p)

        else:

            d=p-boundaries[-1]


            if (
                0.8*period_points
                <
                d
                <
                1.2*period_points
            ):

                boundaries.append(p)



    return np.array(
        boundaries
    )



def save_single_cycles(
        data,
        boundaries,
        output_dir
):


    folder=os.path.join(
        output_dir,
        "单脉冲波形"
    )


    os.makedirs(
        folder,
        exist_ok=True
    )


    total=len(boundaries)-1


    if total<=0:
        return


    ids=[
        0,
        1,
        total//2,
        total-2,
        total-1
    ]


    ids=list(
        dict.fromkeys(ids)
    )


    for i in ids:


        if i<0 or i>=total:
            continue


        cycle=data.iloc[
            boundaries[i]:
            boundaries[i+1]
        ]


        plt.figure(
            figsize=(10,5)
        )


        plt.subplot(
            2,1,1
        )

        plt.plot(
            cycle["time"],
            cycle["voltage"]
        )

        plt.ylabel(
            "Voltage(V)"
        )


        plt.subplot(
            2,1,2
        )

        plt.plot(
            cycle["time"],
            cycle["current"]
        )

        plt.ylabel(
            "Current(A)"
        )

        plt.xlabel(
            "Time(s)"
        )


        plt.tight_layout()


        plt.savefig(
            os.path.join(
                folder,
                f"cycle_{i+1:04d}.png"
            ),
            dpi=300
        )


        plt.close()



def plot_overlay(
        data,
        boundaries,
        output_dir
):


    plt.figure(
        figsize=(8,5)
    )


    for i in range(
        min(20,len(boundaries)-1)
    ):

        cycle=data.iloc[
            boundaries[i]:
            boundaries[i+1]
        ]


        plt.plot(
            cycle["voltage"].values,
            alpha=0.3
        )


    plt.xlabel(
        "Sample"
    )

    plt.ylabel(
        "Voltage(V)"
    )


    plt.title(
        "Cycle Overlay"
    )


    plt.tight_layout()


    plt.savefig(
        os.path.join(
            output_dir,
            "周期叠加验证.png"
        ),
        dpi=300
    )


    plt.close()



def detect_cycles(
        data,
        output_dir="./输出结果"
):


    print("="*70)
    print("自动周期检测")
    print("="*70)


    os.makedirs(
        output_dir,
        exist_ok=True
    )


    time=data["time"].values

    voltage=data["voltage"].values


    dt=np.mean(
        np.diff(time)
    )


    fs=1/dt


    print(
        "采样频率:",
        fs
    )


    period_points=multi_region_period(
        voltage,
        fs
    )


    voltage_low=lowpass_filter(
        voltage,
        fs
    )


    boundaries=locate_boundaries(
        voltage_low,
        period_points
    )


    print(
        "周期边界数量:",
        len(boundaries)
    )


    cycles=[]


    for i in range(
        len(boundaries)-1
    ):


        cycles.append({

            "cycle_id":
                i+1,

            "start_index":
                int(boundaries[i]),

            "end_index":
                int(boundaries[i+1]),

            "duration_us":
                (
                    time[boundaries[i+1]]
                    -
                    time[boundaries[i]]
                )*1e6

        })


    cycle_df=pd.DataFrame(
        cycles
    )


    print(
        "最终周期数量:",
        len(cycle_df)
    )


    if len(cycle_df)>0:

        print(
            "平均周期(us):",
            cycle_df["duration_us"].mean()
        )


        print(
            "标准差(us):",
            cycle_df["duration_us"].std()
        )


    cycle_df.to_csv(
        os.path.join(
            output_dir,
            "周期信息.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )


    save_single_cycles(
        data,
        boundaries,
        output_dir
    )


    plot_overlay(
        data,
        boundaries,
        output_dir
    )


    return cycle_df
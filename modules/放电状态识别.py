import os
import numpy as np
import pandas as pd



def extract_features(cycle):

    voltage = cycle["voltage"].values
    current = cycle["current"].values


    N = len(voltage)



    # ==========================
    # 电压特征
    # ==========================

    Umin = np.min(voltage)

    voltage_std = np.std(voltage)


    voltage_low_ratio = np.sum(
        voltage < 10
    ) / N



    # ==========================
    # 电流基础特征
    # ==========================

    Ipeak = np.max(
        np.abs(current)
    )


    Irms = np.sqrt(
        np.mean(current**2)
    )


    discharge_ratio = np.sum(
        np.abs(current)>3
    ) / N



    energy_J = np.sum(
        np.abs(voltage*current)
    ) / 1e6



    # ==========================
    # 高电流特征
    # ==========================

    I_threshold = max(
        10,
        0.5*Ipeak
    )


    high_index = np.where(
        np.abs(current)>I_threshold
    )[0]



    if len(high_index)>0:

        current_high_ratio = (
            len(high_index)/N
        )


        current_high_std = np.std(
            current[high_index]
        )


        current_high_time = (
            len(high_index)
            /
            1e6
        )


    else:

        current_high_ratio = 0

        current_high_std = 0

        current_high_time = 0



    # ==========================
    # 电流突变特征
    # ==========================


    diff_I = np.diff(
        current
    )


    di_dt_max = np.max(
        np.abs(diff_I)
    ) * 1e6



    I_jump = (
        np.max(current)
        -
        np.mean(current[:int(N*0.1)])
    )



    # ==========================
    # 上升时间
    # ==========================

    try:

        I10 = 0.1*Ipeak
        I90 = 0.9*Ipeak


        idx10 = np.where(
            np.abs(current)>I10
        )[0][0]


        idx90 = np.where(
            np.abs(current)>I90
        )[0][0]


        I_rise_time = (
            idx90-idx10
        )/1e6


    except:

        I_rise_time = 0



    return {

        "Umin":Umin,

        "voltage_low_ratio":
            voltage_low_ratio,

        "voltage_std":
            voltage_std,


        "Ipeak":Ipeak,

        "Irms":Irms,

        "discharge_ratio":
            discharge_ratio,

        "energy_J":
            energy_J,


        "current_high_ratio":
            current_high_ratio,

        "current_high_std":
            current_high_std,

        "current_high_time":
            current_high_time,


        "I_jump":
            I_jump,

        "di_dt_max":
            di_dt_max,

        "I_rise_time":
            I_rise_time

    }





def classify_state(f):


    # ==========================
    # 短路判断
    # ==========================

    if (

        f["voltage_low_ratio"] > 0.5

        and

        f["current_high_ratio"] > 0.5

        and

        f["Irms"] > 20

        and

        f["current_high_std"] < 15

    ):

        return (
            "short_circuit",
            "none"
        )



    # ==========================
    # 开路判断
    # ==========================

    if (

        f["Irms"] < 3

        and

        f["discharge_ratio"] < 0.1

        and

        f["energy_J"] < 0.08

    ):

        return (
            "open_circuit",
            "none"
        )



    # ==========================
    # 有效放电
    # ==========================


    if (

        f["energy_J"] >= 0.08

        or

        f["discharge_ratio"] >=0.1

    ):


        # 异常放电暂时定义

        if (

            f["di_dt_max"]>5000000

            or

            f["current_high_ratio"]>0.5

        ):

            return (
                "effective_discharge",
                "abnormal"
            )

        else:

            return (
                "effective_discharge",
                "normal"
            )


    return (
        "open_circuit",
        "none"
    )





def recognize_discharge_states(
        data,
        cycles,
        output_dir
):


    results=[]


    for _,c in cycles.iterrows():

        start=int(c["start_index"])

        end=int(c["end_index"])


        cycle=data.iloc[start:end]


        f=extract_features(
            cycle
        )


        state,sub=classify_state(
            f
        )


        f["cycle_id"]=c["cycle_id"]

        f["state"]=state

        f["sub_state"]=sub


        results.append(f)



    df=pd.DataFrame(
        results
    )


    os.makedirs(
        output_dir,
        exist_ok=True
    )


    df.to_csv(
        os.path.join(
            output_dir,
            "放电状态结果.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )


    print("\n状态统计:")
    print(
        df["state"].value_counts()
    )


    print("\n子状态统计:")
    print(
        df["sub_state"].value_counts()
    )


    return df
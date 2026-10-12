import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate_short_score(row):

    # 电压低平程度
    U_score = row["voltage_low_ratio"]


    # 高电流持续程度
    I_score = row["current_high_ratio"]


    # 电流大小归一化
    I_score2 = min(
        row["Irms"] / 50,
        1
    )


    # 持续时间
    T_score = min(
        row["current_high_time"] / 0.002,
        1
    )


    # 平顶程度
    if row["Irms"] > 0:

        flat_score = max(
            0,
            1-row["current_high_std"]/row["Irms"]
        )

    else:

        flat_score = 0



    score = (

        0.25*U_score

        +

        0.25*I_score

        +

        0.2*I_score2

        +

        0.15*T_score

        +

        0.15*flat_score

    )


    return score





def short_candidate_analysis(
        result_file,
        output_dir
):


    print("="*70)
    print("短路候选分析")
    print("="*70)


    df = pd.read_csv(
        result_file
    )


    df["short_score"] = df.apply(
        calculate_short_score,
        axis=1
    )


    candidate = df.sort_values(
        "short_score",
        ascending=False
    )


    save_path = os.path.join(
        output_dir,
        "短路候选排序.csv"
    )


    candidate.to_csv(
        save_path,
        index=False,
        encoding="utf-8-sig"
    )


    print(
        "短路评分最高20个周期:"
    )


    print(
        candidate[
            [
                "cycle_id",
                "short_score",
                "Irms",
                "voltage_low_ratio",
                "current_high_ratio",
                "I_jump"
            ]
        ]
        .head(20)
    )


    return candidate.head(20)
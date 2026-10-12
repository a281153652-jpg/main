import os
import pandas as pd



def analyze_features(
        result_file,
        output_dir
):

    print("="*70)
    print("放电状态特征分布分析")
    print("="*70)



    df = pd.read_csv(
        result_file
    )


    print(
        "周期数量:",
        len(df)
    )



    features = [

    "Ipeak",

    "Irms",

    "discharge_ratio",

    "energy_J",

    "voltage_low_ratio",

    "voltage_std",

    "current_high_ratio",

    "current_high_std",

    "current_high_time",

    "I_jump",

    "di_dt_max",

    "I_rise_time"

]



    stat = []



    for f in features:


        if f not in df.columns:

            continue


        data = df[f]


        stat.append({

            "feature":f,

            "min":
                data.min(),

            "25%":
                data.quantile(0.25),

            "50%":
                data.quantile(0.5),

            "75%":
                data.quantile(0.75),

            "90%":
                data.quantile(0.9),

            "95%":
                data.quantile(0.95),

            "max":
                data.max()

        })



    stat_df = pd.DataFrame(
        stat
    )


    stat_df.to_csv(

        os.path.join(
            output_dir,
            "状态特征统计.csv"
        ),

        index=False,

        encoding="utf-8-sig"

    )


    print("\n总体特征分布:")
    print(stat_df)



    print("\n当前分类数量:")

    print(
        df["state"]
        .value_counts()
    )



    print("\n子状态:")

    print(
        df["sub_state"]
        .value_counts()
    )



    # 分状态统计

    print(
        "\n各状态特征均值:"
    )


    mean_df = (

        df
        .groupby("state")
        [features]
        .mean()

    )


    print(
        mean_df
    )


    mean_df.to_csv(

        os.path.join(
            output_dir,
            "各状态特征均值.csv"
        ),

        encoding="utf-8-sig"

    )



    return stat_df
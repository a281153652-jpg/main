import os
import pandas as pd
import matplotlib.pyplot as plt


from modules.data_loader import load_waveform_data
from modules.周期检测 import detect_cycles



def plot_short_candidates(
        input_file,
        score_file,
        output_dir,
        number=10
):


    print("="*70)
    print("绘制短路候选波形")
    print("="*70)



    os.makedirs(
        output_dir,
        exist_ok=True
    )


    # -------------------
    # 读取候选结果
    # -------------------

    candidate = pd.read_csv(
        score_file
    )


    candidate = candidate.head(
        number
    )


    print(
        "绘制周期:"
    )

    print(
        candidate["cycle_id"].values
    )



    # -------------------
    # 读取原始波形
    # -------------------

    data = load_waveform_data(
        input_file
    )


    cycles = detect_cycles(
        data,
        output_dir
    )



    # -------------------
    # 绘图
    # -------------------

    for _,row in candidate.iterrows():


        cycle_id=int(
            row["cycle_id"]
        )


        cycle_info = cycles[
            cycles["cycle_id"]
            ==
            cycle_id
        ]


        if len(cycle_info)==0:
            continue


        start=int(
            cycle_info.iloc[0]["start_index"]
        )

        end=int(
            cycle_info.iloc[0]["end_index"]
        )


        cycle=data.iloc[
            start:end
        ]



        fig,ax1=plt.subplots(
            figsize=(10,5)
        )


        ax1.plot(
            cycle["time"],
            cycle["voltage"],
            label="Voltage"
        )


        ax1.set_ylabel(
            "Voltage(V)"
        )



        ax2=ax1.twinx()


        ax2.plot(
            cycle["time"],
            cycle["current"],
            label="Current"
        )


        ax2.set_ylabel(
            "Current(A)"
        )



        plt.title(
            f"Short candidate cycle {cycle_id}\nscore={row['short_score']:.3f}"
        )


        plt.tight_layout()



        save=os.path.join(
            output_dir,
            f"cycle_{cycle_id}.png"
        )


        plt.savefig(
            save,
            dpi=300
        )


        plt.close()



        print(
            "保存:",
            save
        )
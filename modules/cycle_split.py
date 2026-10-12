import numpy as np



def split_cycles(
        data,
        cycle_length=1000
):


    print("=" * 70)
    print("周期分割")
    print("=" * 70)


    total_points = len(data)


    cycle_number = (
        total_points //
        cycle_length
    )


    cycles=[]


    for i in range(cycle_number):


        start = (
            i *
            cycle_length
        )


        end = (
            start +
            cycle_length
        )


        cycle = (
            data
            .iloc[start:end]
            .copy()
        )


        cycle["cycle_id"] = i+1


        cycles.append(cycle)



    print(
        "周期数量:",
        len(cycles)
    )


    print(
        "单周期点数:",
        cycle_length
    )


    return cycles
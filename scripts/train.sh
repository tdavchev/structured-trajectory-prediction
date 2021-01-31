#!/bin/bash
for name in "hotel" "univ" "ucy" "zara01" "zara02"
do
    declare -a dimensions=(576 720)
    if [ $name = "univ" ]; then
        declare -a dimensions=(480 640)
    fi
    for seed in 0
    do
        dataset_names=""
        if [ $name = "hotel" ]; then
            declare -a dataset_names=("univ" "ucy" "zara01" "zara02")
        elif [ $name = "univ" ]; then
            declare -a dataset_names=("hotel" "ucy" "zara01" "zara02")
        elif [ $name = "ucy" ]; then
            declare -a dataset_names=("hotel" "univ" "zara01" "zara02")
        elif [ $name = "zara01" ]; then
            declare -a dataset_names=("hotel" "univ" "ucy" "zara02")
        elif [ $name = "zara02" ]; then
            declare -a dataset_names=("hotel" "univ" "ucy" "zara01")
        fi

        save_name="map_cond_per_frame_$name"
        writer_name="map_cond_per_frame_$name"
        dynamics_save_name="MDNRNN_RNN_SIZE_768_8_num_mix_5_FLAT_acts_True_3_by_1_$name.json"
        preload=true
        temperature=0.21

        echo "---> dataset names is: ${dataset_names[@]}, save_name: $save_name and dimensions: ${dimensions[@]}, writer_name $writer_name, dynamics_save_name: $dynamics_save_name, preload $preload, tempearature $temperature"
        python scripts/main.py --seed_no $seed --dataset_names ${dataset_names[@]} --dimensions ${dimensions[@]} \
            --save_name $save_name --writer_name $writer_name --dynamics_save_name $dynamics_save_name --preload $preload --temperature $temperature
  done
done

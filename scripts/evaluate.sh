#!/bin/bash
for name in "hotel" "univ" "ucy" "zara01" "zara02"
do
    declare -a dimensions=(576 720)
    if [ $name = "univ" ]; then
        declare -a dimensions=(480 640)
    fi
    echo "dimensions ${dimensions[@]}"
    for temperature in 0.21
    do
        for type in 0 1 # ade or fde
        do
            error_type="ade"
            results_name="ade_results_rd_combined_vae_norm_$name_$temperature"
            if [ $type -eq 1 ]; then
                error_type="fde"
                results_name="fde_results_rd_combined_vae_norm_$name_$temperature"
            fi
            for seed in 0
            do
                for pred in 8 12
                do
                    dataset_names=""
                    if [ $name = "hotel" ]; then
                        declare -a dataset_names=("hotel")
                    elif [ $name = "univ" ]; then
                        declare -a dataset_names=("univ")
                    elif [ $name = "ucy" ]; then
                        declare -a dataset_names=("ucy")
                    elif [ $name = "zara01" ]; then
                        declare -a dataset_names=("zara01")
                    elif [ $name = "zara02" ]; then
                        declare -a dataset_names=("zara02")
                    fi

                    save_name="map_cond_per_frame_$name"
                    writer_name="map_cond_per_frame_$name"
                    dynamics_save_name="MDNRNN_RNN_SIZE_768_8_num_mix_5_FLAT_acts_True_3_by_1_$name.json"
                    preload=0 #true
                    # temperature=0.0001 #0.21
            
                    echo "---> dataset names is: ${dataset_names[@]}, pred: $pred, save_name: $save_name and dimensions: ${dimensions[@]}, writer_name $writer_name, dynamics_save_name: $dynamics_save_name, preload $preload, tempearature $temperature"
                    python scripts/main.py --seed_no $seed --predicted_length $pred --error_type $error_type --results_name $results_name --dataset_names ${dataset_names[@]} --dimensions ${dimensions[@]} \
                        --save_name $save_name --writer_name $writer_name --dynamics_save_name $dynamics_save_name --preload $preload --temperature $temperature
                done
            done
        done
    done
done

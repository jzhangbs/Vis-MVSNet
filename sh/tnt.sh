DIR=<root dir of output result>
for M in <job name>; do
    for S in {0..7}; do
        for NS in 7; do
            echo ${S}_${M}_${NS}
            mkdir -p ${DIR}/depth_${S}_${M}_${NS}
            SCAN=${S} python test.py \
                --data_root <tanksandtemples dir> \
                --dataset_name tanksandtemples \
                --model_name model_cas \
                --num_src ${NS} \
                --max_d 256 \
                --interval_scale 1 \
                --cas_depth_num 64,32,16 \
                --cas_interv_scale 4,2,1 \
                --resize 1920,1080 \
                --crop 1920,1056 \
                --mode soft \
                --load_path <load dir>/${M} \
                --write_result \
                --result_dir ${DIR}/depth_${S}_${M}_${NS}
        done
    done
done
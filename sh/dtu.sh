L_SCAN=( 24 37 40 55 65 69 83 97 105 106 110 114 118 122 )
L_JOB=( vis )
L_MODE=( soft )
L_NS=( 3 )
DIR=~/code/holistic_diffusion/logs/dtu/vis

for ((i=0;i<${#L_JOB[@]};i++)); do
    JOB=${L_JOB[i]}
    MODE=${L_MODE[i]}
    for NS in ${L_NS[@]}; do
        for SCAN in ${L_SCAN[@]}; do
            echo SCAN ${SCAN} JOB ${JOB} MODE ${MODE} NS ${NS}
            mkdir -p ${DIR}/depth_${SCAN}_${JOB}_${NS}
            SCAN=${SCAN} python test.py \
                --data_root /mnt/data/aspen_dtu_o-1.0.0/data/raw \
                --dataset_name dtu_test \
                --model_name model_cas \
                --num_src ${NS} \
                --max_d 256 \
                --interval_scale 0.75 \
                --cas_depth_num 64,32,16 \
                --cas_interv_scale 4,2,1 \
                --crop 1200,1200 \
                --resize 1024,1024 \
                --mode ${MODE} \
                --load_path pretrained_model/${JOB} \
                --write_result \
                --result_dir ${DIR}/depth_${SCAN}_${JOB}_${NS}
        done
    done
done

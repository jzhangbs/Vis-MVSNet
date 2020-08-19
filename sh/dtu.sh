L_SCAN=$(seq 0 21)
L_JOB=( <job name> )
L_MODE=( soft )
L_NS=( 5 )
DIR=<root dir of output result>

for ((i=0;i<${#L_JOB[@]};i++)); do
    JOB=${L_JOB[i]}
    MODE=${L_MODE[i]}
    for NS in ${L_NS[@]}; do
        for SCAN in ${L_SCAN[@]}; do
            echo SCAN ${SCAN} JOB ${JOB} MODE ${MODE} NS ${NS}
            mkdir -p ${DIR}/depth_${SCAN}_${JOB}_${NS}
            SCAN=${SCAN} python test.py \
                --data_root <dtu dir> \
                --dataset_name dtu_test \
                --model_name model_cas \
                --num_src ${NS} \
                --max_d 256 \
                --interval_scale 0.75 \
                --cas_depth_num 64,32,16 \
                --cas_interv_scale 4,2,1 \
                --resize 1600,1200 \
                --crop 1600,1184 \
                --mode ${MODE} \
                --load_path <load dir>/${JOB} \
                --write_result \
                --result_dir ${DIR}/depth_${SCAN}_${JOB}_${NS}
        done
    done
done

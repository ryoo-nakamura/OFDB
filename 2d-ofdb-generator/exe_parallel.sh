#!/bin/bash

numof_category=1000
fillrate=0.2
weight=0.4
imagesize=362
numof_point=100000
numof_ite=200000
howto_draw='patch_gray'
numof_thread=40


# Parameter search
python param_search/ifs_search.py --rate=${fillrate} --category=${numof_category} --numof_point=${numof_point} --save_dir='../dataset'
python param_search/parallel_dir.py --path2dir='../dataset/' --rate=${fillrate} --category=${numof_category} --thread=${numof_thread}

# Multi-thread processing
SECONDS=0
for ((i=0 ; i<${numof_thread} ; i++))
do
    python fractal_renderer/make_2dofdb.py \
        --load_root='../dataset/csv_rate'${fillrate}'_category'${numof_category}'/csv'${i} \
        --save_root='../dataset/2d-ofdb-'${numof_category} --image_size_x=${imagesize} --image_size_y=${imagesize} \
        --iteration=${numof_ite} --draw_type=${howto_draw} --weight_csv='./fractal_renderer/weights/weights_'${weight}'.csv' &
done
wait
run_time=$SECONDS
echo $run_time
echo "End generate images process"

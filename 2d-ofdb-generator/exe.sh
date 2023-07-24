
numof_category=10
fillrate=0.2
weight=0.4
imagesize=10000
numof_point=100000
howto_draw='patch_gray'


# Parameter search
python param_search/ifs_search.py --rate=${fillrate} --category=${numof_category} --numof_point=${numof_point} --save_dir='../dataset'

# Create FractalDB
python fractal_renderer/make_2dofdb.py \
        --load_root='../dataset/csv_rate'${fillrate}'_category'${numof_category} \
        --save_root='../dataset/2d-ofdb-'${numof_category} --image_size_x=${imagesize} --image_size_y=${imagesize} \
        --iteration=${numof_ite} --draw_type=${howto_draw} --weight_csv='./fractal_renderer/weights/weights_'${weight}'.csv'

# !/bin/bash

variance_threshold=0.05
numof_category=1000
param_path='../dataset/3D-OFDB-'${numof_category}'/3DIFS_params'
model_save_path='../dataset/3D-OFDB-'${numof_category}'/3Dmodels'
image_save_path='../dataset/3D-OFDB-'${numof_category}'/image'

#### Parameter search
SECONDS=0
python 3dfractal_render/category_search.py --variance=${variance_threshold} --numof_classes=${numof_category} --save_root=${param_path}
run_time=$SECONDS
echo $run_time
echo "End Parameter search process"

# Generate 3D fractal model
SECONDS=0
python 3dfractal_render/instance.py --load_root ${param_path} --save_root ${model_save_path} --classes ${numof_category}
run_time=$SECONDS
echo $run_time
echo "End Generate 3D fractal model process"


# Render Multi-view images
export PYOPENGL_PLATFORM=egl
SECONDS=0
python image_render/render1view.py --load_root ${model_save_path} --save_root ${image_save_path} --view_point 12
run_time=$SECONDS
echo $run_time
echo "End Render Multi-view images process"


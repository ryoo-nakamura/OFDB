#!/bin/sh

source /etc/profile.d/modules.sh
module load gcc/12.2.0
module load python/3.10/3.10.10
module load cuda/10.2/10.2.89
module load cudnn/7.6/7.6.5
module load nccl/2.7/2.7.8-1

export PATH="/home/aab10659de/anaconda3/bin:${PATH}"
source activate nakamura_vit

export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep inet | cut -d " " -f 6 | cut -d "/" -f 1)
export NGPUS=8
export NPERNODE=4

# cd /groups/gaa50131/user/nakamura/Fractal_script/ImageClassification
# wandb login 1f6461febebe3a677cd0465561652aecd814b0e0


out_dir=outputs/3d-ofdb

mpiexec -npernode ${NPERNODE} -np ${NGPUS} python -B main.py data=grayimagefolder \
                data.baseinfo.name=3d-ofdb-1k \
                data.baseinfo.train_imgs=1000 data.baseinfo.num_classes=1000 \
                data.trainset.root=../../../../../../dataset/3D-OFDB-1000/image \
                data.loader.batch_size=32 data.transform.no_aug=False data.transform.auto_augment=rand-m9-mstd0.5-inc1 \
                data.transform.re_prob=0.5 data.transform.color_jitter=0.4 data.transform.hflip=0.5 data.transform.vflip=0.5 \
                data.transform.scale=[0.08,1.0] data.transform.ratio=[0.75,1.3333] data.mixup.prob=1.0 data.mixup.mixup_alpha=0.8 \
                data.mixup.cutmix_alpha=1.0 data.mixup.switch_prob=0.5 model=vit \
                model.arch.model_name=vit_tiny_patch16_224 model.optim.learning_rate=0.001 \
                model.scheduler.args.warmup_epochs=5 logger.group=cpmpare_instance_augmentation \
                logger.save_epoch_freq=10000 epochs=80000 mode=pretrain \
                output_dir=${out_dir}

mpiexec -npernode ${NPERNODE} -np ${NGPUS} python -B main.py data=grayimagefolder data.baseinfo.name=fractal1k_ins_patch0_folder data.baseinfo.train_imgs=1000 data.baseinfo.num_classes=1000 data.trainset.root=/groups/gaa50131/user/nakamura/FractalDB-Pretrained-ResNet-PyTorch/data/FractalDB-1000-tadokoro_fill data.loader.batch_size=32 data.transform.no_aug=False data.transform.auto_augment=rand-m9-mstd0.5-inc1 data.transform.re_prob=0.5 data.transform.color_jitter=0.4 data.transform.hflip=0.5 data.transform.vflip=0.5 data.transform.scale=[0.08,1.0] data.transform.ratio=[0.75,1.3333] data.mixup.prob=1.0 data.mixup.mixup_alpha=0.8 data.mixup.cutmix_alpha=1.0 data.mixup.switch_prob=0.5 model=vit model.arch.model_name=vit_tiny_patch16_224 model.optim.learning_rate=0.001 model.scheduler.args.warmup_epochs=5 logger.group=cpmpare_instance_augmentation logger.save_epoch_freq=1000 epochs=80000 mode=pretrain output_dir=soutputs/cpmpare_instance_augmentation/vit_tiny_patch16_224/fractal1k_ins_patch0_folder/pretrain/0930_003820 
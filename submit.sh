#!/bin/sh

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# This is an explanatory file for how to run our code on a SLURM cluster. It is not
# meant to be run out of the box but should be useful if this is your use case.

timestamp="$(date +"%Y-%m-%d--%H:%M:%S")"
mkdir /checkpoint/ow045820/code/$timestamp/
rsync -r --exclude '.ipynb_checkpoints/'  --exclude '.vscode/'  \
         --exclude 'mp3dtatarchenko' --exclude 'mp3dviewapp' --exclude 'mp3ddeepvoxelsunet' --exclude 'mp3dzbufferpts' \
         --exclude 'method_figure' --exclude 'real_estate_results' --exclude 'checkpoints' --exclude 'modelcheckpoints' \
         --exclude 'data/.ipynb_checkpoints'  --exclude 'temp/' --exclude 'data_preprocessing/temp/'   \
         --exclude 'results/'  --exclude '.git/' ~/projects/code/synsin_public /checkpoint/ow045820/code/$timestamp/
rsync -r ~/projects/code/torch3d_fork/torch3d /checkpoint/ow045820/code/$timestamp/

cd /checkpoint/ow045820/code/$timestamp/synsin_public/

TIMESTAMP=timestamp

chmod +x ./submit_slurm_synsin.sh

radius=4
radpow=2

#################################################################################################
### Setting up parameters for different datasets
#################################################################################################

# KITTI
#additionalops="--num_workers 10 --dataset kitti --use_inv_z --lr 0.0001 --use_inverse_depth"

# Matterport
additionalops=" --num_workers 0  --lr 0.0001 "

# RealEstate10K
#additionalops=" --num_workers 10 --dataset realestate --use_inv_z  --lr 0.0001 "


#################################################################################################
### Ablation for different hyperparameters when compositing
#################################################################################################

#pppixel=4
#modeltype='zbuffer_pts'
#accumulation="alphacomposite"
#refinemodeltype='resnet_256W8UpDown64'
#suffix="_accum${accumulation}_${radius}_${radpow}_redo_ppp4"
#sbatch --export=ALL,radpow=$radpow,accumulation=$accumulation,radius=$radius,additionalops="$additionalops",pppixel=$pppixel,taugumbel=$taugumbel,modeltype=$modeltype,refinemodeltype=$refinemodeltype,suffix=$suffix ./submit_slurm_synsin.sh

# pppixel=8
# modeltype='zbuffer_pts'
# accumulation="alphacomposite"
# refinemodeltype='resnet_256W8UpDown64'
# suffix="_accum${accumulation}_${radius}_${radpow}_redo_ppp8_maxdisparity"
# sbatch --export=ALL,radpow=$radpow,accumulation=$accumulation,radius=$radius,additionalops="$additionalops",pppixel=$pppixel,taugumbel=$taugumbel,modeltype=$modeltype,refinemodeltype=$refinemodeltype,suffix=$suffix ./submit_slurm_synsin.sh

pppixel=128
modeltype='zbuffer_pts'
accumulation="alphacomposite"
refinemodeltype='resnet_256W8UpDown64'
suffix="_accum${accumulation}_${radius}_${radpow}_redo_maxdisparity"
sbatch --export=ALL,radpow=$radpow,accumulation=$accumulation,radius=$radius,additionalops="$additionalops",pppixel=$pppixel,taugumbel=$taugumbel,modeltype=$modeltype,refinemodeltype=$refinemodeltype,suffix=$suffix ./submit_slurm_synsin.sh



##pppixel=1
#modeltype='zbuffer_pts'
#accumulation="alphacomposite"
#refinemodeltype='resnet_256W8UpDown64'
#suffix="_accum${accumulation}_${radius}_${radpow}_ppp${ppppixel}_redo"
#sbatch --export=ALL,radpow=$radpow,accumulation=$accumulation,radius=$radius,additionalops="$additionalops",pppixel=$pppixel,taugumbel=$taugumbel,modeltype=$modeltype,refinemodeltype=$refinemodeltype,suffix=$suffix ./submit_slurm_synsin.sh


#pppixel=128
#radius=0.5
#modeltype='zbuffer_pts'
#accumulation="alphacomposite"
#refinemodeltype='resnet_256W8UpDown64'
#suffix="_accum${accumulation}_${radius}_${radpow}_redo"
#sbatch --export=ALL,radpow=$radpow,accumulation=$accumulation,radius=$radius,additionalops="$additionalops",pppixel=$pppixel,taugumbel=$taugumbel,modeltype=$modeltype,refinemodeltype=$refinemodeltype,suffix=$suffix ./submit_slurm_synsin.sh


#pppixel=128
#radius=4
#modeltype='zbuffer_pts'
#accumulation="alphacomposite"
#refinemodeltype='resnet_256W8UpDown3'
#suffix="_accum${accumulation}_${radius}_${radpow}_rgb_redo"
#additionalops="$additionalops --use_rgb_features"
#sbatch --export=ALL,radpow=$radpow,accumulation=$accumulation,radius=$radius,additionalops="$additionalops",pppixel=$pppixel,taugumbel=$taugumbel,modeltype=$modeltype,refinemodeltype=$refinemodeltype,suffix=$suffix ./submit_slurm_synsin.sh

#pppixel=128
#radius=4
#modeltype='zbuffer_pts'
#accumulation="alphacomposite"
#refinemodeltype='resnet_256W8UpDown64'
#suffix="_accum${accumulation}_${radius}_${radpow}_gtdepth_redo"
#additionalops="$additonalops --use_gt_depth"
#sbatch --export=ALL,radpow=$radpow,accumulation=$accumulation,radius=$radius,additionalops="$additionalops",pppixel=$pppixel,taugumbel=$taugumbel,modeltype=$modeltype,refinemodeltype=$refinemodeltype,suffix=$suffix ./submit_slurm_synsin.sh

#pppixel=128
#radius=4
#modeltype='zbuffer_pts'
#accumulation="alphacomposite"
#refinemodeltype='resnet_256W8UpDown64'
#suffix="_accum${accumulation}_${radius}_${radpow}_traindepth_redo"
#additionalops="$additonalops --train_depth"
#sbatch --export=ALL,radpow=$radpow,accumulation=$accumulation,radius=$radius,additionalops="$additionalops",pppixel=$pppixel,taugumbel=$taugumbel,modeltype=$modeltype,refinemodeltype=$refinemodeltype,suffix=$suffix ./submit_slurm_synsin.sh

#################################################################################################
### Different accumulation Functions
#################################################################################################


# pppixel=128
# modeltype='zbuffer_pts'
# accumulation="wsum"
# refinemodeltype='resnet_256W8UpDown64'
# suffix="_accum${accumulation}_${radius}_${radpow}_redo"
# additionalops=""
# sbatch --export=ALL,radpow=$radpow,accumulation=$accumulation,radius=$radius,additionalops="$additionalops",pppixel=$pppixel,taugumbel=$taugumbel,modeltype=$modeltype,refinemodeltype=$refinemodeltype,suffix=$suffix ./submit_slurm_synsin.sh

# pppixel=128
# modeltype='zbuffer_pts'
# accumulation="wsumnorm"
# refinemodeltype='resnet_256W8UpDown64'
# suffix="_accum${accumulation}_${radius}_${radpow}_redo"
# additionalops=""
# sbatch --export=ALL,radpow=$radpow,accumulation=$accumulation,radius=$radius,additionalops="$additionalops",pppixel=$pppixel,taugumbel=$taugumbel,modeltype=$modeltype,refinemodeltype=$refinemodeltype,suffix=$suffix ./submit_slurm_synsin.sh

# pppixel=128
# modeltype='zbuffer_pts'
# accumulation="alphacomposite"
# refinemodeltype='resnet_256W8UpDown64'
# suffix="_accum${accumulation}_${radius}_${radpow}_redo"
# additionalops=""
# sbatch --export=ALL,radpow=$radpow,accumulation=$accumulation,radius=$radius,additionalops="$additionalops",pppixel=$pppixel,taugumbel=$taugumbel,modeltype=$modeltype,refinemodeltype=$refinemodeltype,suffix=$suffix ./submit_slurm_synsin.sh



#! /bin/bash

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# This is an explanatory file for how to run our code on a SLURM cluster. It is not
# meant to be run out of the box but should be useful if this is your use case.

#################################################################################
#     File Name           :     submit_synsin.sh
#     Created By          :     Olivia Wiles
#     Description         :     submit view synthesis jobs 
#################################################################################

#SBATCH --job-name=vs3d

#SBATCH --output=/checkpoint/%u/jobs/sample-%j.out

#SBATCH --error=/checkpoint/%u/jobs/sample-%j.err

#SBATCH --nodes=1 -C volta32gb

#SBATCH --partition=dev

#SBATCH --ntasks-per-node=1

#SBATCH --gres=gpu:volta:4

#SBATCH --cpus-per-task=40

#SBATCH --mem=250G

#SBATCH --signal=USR1@600

#SBATCH --open-mode=append

#SBATCH --time=72:00:00

# The ENV below are only used in distributed training with env:// initialization
export MASTER_ADDR=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:3}
export MASTER_PORT=29500 

unset PYTHONPATH 
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvidia-opengl:${LD_LIBRARY_PATH}
export USE_SLURM=1

source activate synsin_env

export DEBUG=0
echo $additionalops

echo "Starting model with... radius $radius^$radpow pp $pppixel  with model type $modeltype, accumulation $accumulation and $refinemodeltype saving to $suffix"
srun --label python train.py --batch-size 32 --folder 'final2' \
        --pp_pixel $pppixel --radius $radius --resume --accumulation $accumulation --rad_pow $radpow  \
        --model_type $modeltype --refine_model_type $refinemodeltype $additionalops \
        --norm_G 'sync:spectral_batch' --gpu_ids 0,1,2 --render_ids 3 \
        --suffix $suffix --normalize_image #--W 512 \

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

source activate synsin_env

export DEBUG=0
export USE_SLURM=0

# How to run on RealEstate10K
python train.py --batch-size 32 --folder 'temp' --num_workers 4  \
         --resume --dataset 'realestate' --use_inv_z --accumulation 'alphacomposite' \
         --model_type 'zbuffer_pts' --refine_model_type 'resnet_256W8UpDown64'  \
         --norm_G 'sync:spectral_batch' --gpu_ids 0,1 --render_ids 1 \
         --suffix '' --normalize_image --lr 0.0001

# How to run on KITTI
# python train.py --batch-size 32 --folder 'temp' --num_workers 4  \
#         --resume --dataset 'kitti' --use_inv_z --use_inverse_depth --accumulation 'alphacomposite' \
#         --model_type 'zbuffer_pts' --refine_model_type 'resnet_256W8UpDown64'  \
#         --norm_G 'sync:spectral_batch' --gpu_ids 0,1 --render_ids 1 \
#         --suffix '' --normalize_image --lr 0.0001

# # How to run on Matterport3D
# python train.py --batch-size 32 --folder 'temp' --num_workers 0  \
#        --resume --accumulation 'alphacomposite' \
#        --model_type 'zbuffer_pts' --refine_model_type 'resnet_256W8UpDown64'  \
#        --norm_G 'sync:spectral_batch' --gpu_ids 0 --render_ids 1 \
#        --suffix '' --normalize_image --lr 0.0001

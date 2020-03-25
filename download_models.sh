# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# BASH file for downloading pretrained models.

# Download readme
mkdir ./modelcheckpoints/
cd modelcheckpoints/
wget https://dl.fbaipublicfiles.com/synsin/checkpoints/readme.txt

# Make dataset directories
mkdir kitti
mkdir realestate
mkdir mp3d

# Download files
cd ./kitti/
wget https://dl.fbaipublicfiles.com/synsin/checkpoints/kitti/zbufferpts_invdepth.pth
wget https://dl.fbaipublicfiles.com/synsin/checkpoints/kitti/synsin_invdepth.pth

wget https://dl.fbaipublicfiles.com/synsin/checkpoints/kitti/viewappearance.pth
wget https://dl.fbaipublicfiles.com/synsin/checkpoints/kitti/tatarchenko.pth

cd ../realestate/
wget https://dl.fbaipublicfiles.com/synsin/checkpoints/realestate/synsin.pth
wget https://dl.fbaipublicfiles.com/synsin/checkpoints/realestate/zbufferpts.pth

wget https://dl.fbaipublicfiles.com/synsin/checkpoints/realestate/viewappearance.pth
wget https://dl.fbaipublicfiles.com/synsin/checkpoints/realestate/tatarchenko.pth

cd ../mp3d/
wget https://dl.fbaipublicfiles.com/synsin/checkpoints/mp3d/synsin.pth
wget https://dl.fbaipublicfiles.com/synsin/checkpoints/mp3d/viewappearance.pth
wget https://dl.fbaipublicfiles.com/synsin/checkpoints/mp3d/tatarchenko.pth

cd ../

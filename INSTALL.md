
<h1> Setup and Installation </h1>

Install the requirements. 
- These are located in `./requirements.txt`. 
- They can be installed with `conda create –-name synsin_env –-file requirements.txt`. (Note that this file is missing the tensorboardX dependency which needs to be installed separately if you wish to train a model yourself. It is not necessary for running the demos or evaluation.)

Or install the requirements yourself. Requirements:
- pytorch=1.4
- torchvision=0.5
- opencv-python (pip)
- [pytorch3d](https://github.com/facebookresearch/pytorch3d). Make sure it's the most recent version,
as the older version does not have the required libraries.
- tensorboardX (pip)
- jupyter; matplotlib
- [habitat-api](https://github.com/facebookresearch/habitat-api) (if you want to use Matterport or Replica datasets)
- [habitat-sim](https://github.com/facebookresearch/habitat-sim) (if you want to use Matterport or Replica datasets)

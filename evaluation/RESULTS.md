<h1> Results for released models </h1>

Note that this repository is a large refactoring of the original code to allow for public release
and to integrate with [pytorch3d](https://github.com/facebookresearch/pytorch3d).
Hence the models/datasets are not necessarily the same as that in the paper, as we cannot release
the saved test images we used.
To compare results, we recommend comparing against the numbers and models in this repo for fair comparison
and reproducibility.

These models have been trained with the same learning rate (0.0001) and number of epochs.
If you want to train the baselines differently (for example in the paper we found that the voxel based methods were highly sensitive to learning rates, so used a model trained with a lower learning rate), look at the options in `./../submit.sh`.

You can use these numbers to:
1. Compare against your models
2. Verify your setup is indeed correct

<h2> Results on RealEstate </h2>


|                           | PSNR          | SSIM  | Perc SIM      |
| --------------------------|:-------------:| -----:|:-------------:|
| SynSin                    |     22.31     | 0.74  |     1.18      |
| SynSin<sup>+<sup>         |     22.83     | 0.75  |     1.13      |
| ViewAppearance [1]        |     17.05     | 0.56  |     2.19      |
| Tatarchenko [2]           |     11.35     | 0.33  |     3.95      |
| StereoMag [4]             |     25.34     | 0.82  |     1.19      |
| 3DPaper [5]               |     21.88     | 0.66  |     1.52      |

<h2> Results on Matterport </h2>

|                          | PSNR          | SSIM  | Perc SIM      |
| -------------------------|:-------------:| -----:|:-------------:|
| SynSin                   |     20.91     | 0.72  |    1.68       |
| ViewAppearance [1]       |     15.87     | 0.53  |    2.99       |
| Tatarchenko [2]          |     14.79     | 0.57  |    3.73       |

<h2> Results on Replica </h2>

|                          | PSNR          | SSIM  | Perc SIM      |
| -------------------------|:-------------:| -----:|:-------------:|
| SynSin                   |     21.94     | 0.81  |     1.55      |
| ViewAppearance [1]       |     17.42     | 0.66  |     2.29      |
| Tatarchenko [2]          |     14.36     | 0.68  |     3.36      |

<h2> Results on KITTI </h2>

|                           | PSNR          | SSIM  | Perc SIM      |
| --------------------------|:-------------:| -----:|:-------------:|
| SynSin<sup>*<sup>         |    16.70      | 0.52  |    2.07       |
| SynSin<sup>+*<sup>        |    16.73      | 0.52  |    2.05       |
| ViewAppearance [1]        |    14.21      | 0.43  |    2.51       |
| Tatarchenko [2]           |    10.31      | 0.30  |    3.48       |
| ContView [3]              |    16.90      | 0.54  |    2.21       |


<h2> References </h2>
The implemented models are based on:

<sup>*</sup>: Using inverse depth as opposed to a uniform sampling. This is better if there is a long tail distribution of the true depths (as in the KITTI case).

<sup>+</sup>: Leaving the model to run for longer than the paper for a small boost in results.

[1] Zhou, Tinghui, et al. "View synthesis by appearance flow." ECCV, 2016.

[2] Dosovitskiy, Alexey, et al. "Learning to generate chairs with convolutional neural networks." CVPR, 2015.

[3] Chen, Xu, et al. "Monocular Neural Image Based Rendering with Continuous View Control." ICCV, 2019.

[4] Zhou, Tinghui, et al. "Stereo Magnification: Learning View Synthesis using Multiplane Images." SIGGRAPH, 2018.

[5] Code based on work by folks at Facebook. The code used is an early version of the [3D Photos work](https://ai.facebook.com/blog/-powered-by-ai-turning-any-2d-photo-into-3d-using-convolutional-neural-nets/).

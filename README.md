<h1> SynSin: End-to-end View Synthesis from a Single Image (CVPR 2020) </h1>

This is the code for the [CVPR 2020 paper](https://arxiv.org/abs/1912.08804).
This code allows for synthesising of new views of a scene given a single image of an <em>unseen</em> scene at test time.
It is trained with pairs of views in a self-supervised fashion.
It is trained end to end, using GAN techniques and a new differentiable point cloud renderer.
At test time, a single image of an unseen scene is input to the model from which new views are generated.

<p align="center">
  <img width='150' height='150' src="http://www.robots.ox.ac.uk/~ow/videos/media24.gif"/> 
  <img width='150' height='150' src="http://www.robots.ox.ac.uk/~ow/videos/media2.gif"/> 
  <img width='150' height='150' src="http://www.robots.ox.ac.uk/~ow/videos/media18.gif"/> 
  <img width='150' height='150' src="http://www.robots.ox.ac.uk/~ow/videos/media9.gif"/> 
  <img width='150' height='150' src="http://www.robots.ox.ac.uk/~ow/videos/media5.gif"/> 
     
   **Fig 1: Generated images at new viewpoints using SynSin.** Given the first image in the video, the model generates all subsequent images along the trajectory. The same model is used for all reconstructions. The scenes were not seen at train time.
</p>

# Usage

Note that this repository is a large refactoring of the original code to allow for public release
and to integrate with [pytorch3d](https://github.com/facebookresearch/pytorch3d).
Hence the models/datasets are not necessarily the same as that in the paper, as we cannot release
the saved test images we used.
To compare results, we recommend comparing against the numbers and models in this [repo](./evaluation/RESULTS.md) for fair comparison
and reproducibility.

## Setup and Installation 
See [INSTALL](./INSTALL.md).

## Quickstart 
To quickly start using a pretrained model, see [Quickstart](./QUICKSTART.md).

<h2> Training and evaluating your own model </h2>

To download, train, or evaluate a model on a given dataset, please read the appropriate file.
(Note that we cannot distribute the raw pixels, so we have explained how we downloaded and organised the datasets in the appropriate file.)

- [RealEstate10K](./REALESTATE.md)
- [MP3D and Replica](./MP3D.md)
- [KITTI](./KITTI.md)

<h1>Citation</h1>
If this work is helpful in your research. Please cite:

```
@inproceedings{wiles2020synsin,
  author =       {Olivia Wiles and Georgia Gkioxari and Richard Szeliski and 
                  Justin Johnson},
  title =        {{SynSin}: {E}nd-to-end View Synthesis from a Single Image},
  booktitle =      {CVPR},
  year =         {2020}
}
```

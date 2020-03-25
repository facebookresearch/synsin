<h1>KITTI</h1>

<h2>Download</h2>

Download from [continuous_view_synthesis](https://github.com/xuchen-ethz/continuous_view_synthesis).
Store the files in `${KITTI_HOME}/dataset_kitti`.

## Train

### Update options

Update the paths in `./options/options.py` for the dataset being used.

### Training scripts
Use the `./train.sh` to train one of the models on a single GPU node.

You can also look at `./submit_slurm_synsin.sh` to see how to modify parameters in the renderer
and run on a slurm cluster.

## Evaluate

To evaluate, we run the following script. This gives us a bunch of generated vs ground truth images. 

```bash
export KITTI=${KITTI_HOME}/dataset_kitti/images
python evaluation/eval_kitti.py --old_model ${OLD_MODEL} --result_folder ${TEST_FOLDER}
```

We then compare the generated to ground truth images using the evaluation script.

```bash
python evaluation/evaluate_perceptualsim.py \
     --folder ${TEST_FOLDER} \
     --pred_image im_B.png \
     --target_image im_res.png \
     --output_file kitti_results
```

The results we get for each model is given in [RESULTS.md](https://github.com/fairinternal/synsin_public/tree/master/evaluation/RESULTS.md).

If you do not get approximately the same results (some models use noise as input, so there is some randomness), then there is probably an error in your setup:
- Check the libraries.
- Check the data setup is indeed correct.

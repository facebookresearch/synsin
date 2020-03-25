<h1>MP3D and Replica</h1>

<h2>Download</h2>

<h3>Matterport3D</h3>

1. Download habitat:
- [habitat-api](https://github.com/facebookresearch/habitat-api) (if you want to use Matterport or Replica datasets)
- [habitat-sim](https://github.com/facebookresearch/habitat-sim) (if you want to use Matterport or Replica datasets)

2. Download the [point nav datasets](https://github.com/facebookresearch/habitat-api#task-datasets).

3. Download [MP3D](https://niessner.github.io/Matterport/). 

<h3>Replica</h3>

Do the steps for Matterport.

Download [Replica](https://github.com/facebookresearch/Replica-Dataset).


## Train

### Update options

Update the paths in `./options/options.py` for the dataset being used.

### Training scripts
Use the `./train.sh` to train one of the models on a single GPU node.

You can also look at `./submit_slurm_synsin.sh` to see how to modify parameters in the renderer
and run on a slurm cluster.

## Evaluate

Run the evaluation to obtain both visibility and invisibility scores. 

Run the following bash command. It will output some sample images, and save the results to a txt file. Make sure to set the options correctly, else this will throw an error, as the results won't be compatible with our given results.

```bash
python evaluation/eval.py \
     --result_folder ${TEST_FOLDER} \
     --old_model ${OLD_MODEL} \
     --batch_size 8 --num_workers 10  --images_before_reset 200 \ # It is IMPORTANT to set these correctly
     --dataset replica # ONLY if you want to evaluate on replica
```

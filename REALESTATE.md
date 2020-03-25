<h1>RealEstate</h1>

<h2>Download</h2>

Download from [RealEstate10K](https://google.github.io/realestate10k/).
Store the files in the following structure. The `${REAL_ESTATE_10K}/test/` and `${REAL_ESTATE_10K}/train` folders store the original text files.

The frames need to be extracted based on the text files; we extract them to: `${REAL_ESTATE_10K}/frames`. There may be some missing videos, so we use some additional files as described below.

We use a file `${REAL_ESTATE_10K}/frames/train/video_loc.txt` and `${REAL_ESTATE_10K}/frames/test/video_loc.txt` to store the location of the extracted videos. Finally, for each extracted video located at `${REAL_ESTATE_10K}/frames/train/${path_totrain_vid1}/*.png`, we create a new text file `${REAL_ESTATE_10K}/frames/train/${path_totrain_vid1}.txt` which stores the metadata for each frame (this is necessary as there may be some errors in the extraction process). The `${REAL_ESTATE_10K}/frames/train/${path_totrain_vid1}.txt` file is in the same structure as the original text file, except all rows containing images that were not extracted, have been removed.

After following the above, you should have the following structure:

```bash
- ${REAL_ESTATE_10K}/test/*.txt

- ${REAL_ESTATE_10K}/train/*.txt

- ${REAL_ESTATE_10K}/frames/train/
- ${REAL_ESTATE_10K}/frames/train/video_loc.txt
- ${REAL_ESTATE_10K}/frames/train/${path_totrain_vid1}/*.png
- ${REAL_ESTATE_10K}/frames/train/${path_totrain_vid1}.txt
...
- ${REAL_ESTATE_10K}/frames/train/${path_totrain_vidN}/*.png
- ${REAL_ESTATE_10K}/frames/train/${path_totrain_vidN}.txt

- ${REAL_ESTATE_10K}/frames/test/
- ${REAL_ESTATE_10K}/frames/test/video_loc.txt
- ${REAL_ESTATE_10K}/frames/test/${path_totest_vid1}/*.png
- ${REAL_ESTATE_10K}/frames/test/${path_totest_vid1}.txt
...
- ${REAL_ESTATE_10K}/frames/test/${path_totest_vidN}/*.png
- ${REAL_ESTATE_10K}/frames/test/${path_totest_vidN}.txt
```

where `${REAL_ESTATE_10K}/frames/train/video_loc.txt` contains:

```bash
${path_totrain_vid1}
...
${path_totrain_vidN}
```

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
export REALESTATE=${REAL_ESTATE_10K}/frames/test/ 
python evaluation/eval_realestate.py --old_model ${OLD_MODEL} --result_folder ${TEST_FOLDER} 
```

We then compare the generated to ground truth images using the evaluation script.

```bash
python evaluation/evaluate_perceptualsim.py \
     --folder ${TEST_FOLDER} \
     --pred_image output_image_.png \
     --target_image tgt_image_.png \
     --output_file ${TEST_FOLDER}/realestate_results \
     --take_every_other # Used for RealEstate10K when comparing to methods that uses 2 images per output
```

The results we get for each model is given in [RESULTS.md](https://github.com/fairinternal/synsin_public/tree/master/evaluation/RESULTS.md).

If you do not get approximately the same results (some models use noise as input, so there is some randomness), then there is probably an error in your setup:
- Check the libraries.
- Check the data setup is indeed correct.


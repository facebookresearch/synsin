# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import json
import os
import time

import cv2
import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
from torch.multiprocessing import set_start_method
from torch.utils.data import DataLoader

from evaluation.metrics import perceptual_sim, psnr, ssim_metric

from models.base_model import BaseModel
from models.depth_model import Model
from models.networks.pretrained_networks import PNet
from models.networks.sync_batchnorm import convert_model

from options.options import get_dataset, get_model
from options.test_options import ArgumentParser
from utils.geometry import get_deltas

torch.manual_seed(0)
torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id):
    torch.manual_seed(worker_id)


def psnr_mask(pred_imgs, key, invis=False):
    mask = pred_imgs["OutputImg"] == pred_imgs["SampledImg"]
    mask = mask.float().min(dim=1, keepdim=True)[0]

    if invis:
        mask = 1 - mask

    return psnr(pred_imgs["OutputImg"], pred_imgs[key], mask)


def ssim_mask(pred_imgs, key, invis=False):
    mask = pred_imgs["OutputImg"] == pred_imgs["SampledImg"]
    mask = mask.float().min(dim=1, keepdim=True)[0]

    if invis:
        mask = 1 - mask

    return ssim_metric(pred_imgs["OutputImg"], pred_imgs[key], mask)


def perceptual_sim_mask(pred_imgs, key, vgg16, invis=False):
    mask = pred_imgs["OutputImg"] == pred_imgs["SampledImg"]
    mask = mask.float().min(dim=1, keepdim=True)[0]

    if invis:
        mask = 1 - mask

    return perceptual_sim(
        pred_imgs["OutputImg"] * mask, pred_imgs[key] * mask, vgg16
    )

def check_initial_batch(batch, dataset):
    try:
        if dataset == 'replica':
            np.testing.assert_allclose(batch['cameras'][0]['P'].data.numpy().ravel(), 
                np.loadtxt('./data/files/eval_cached_cameras_replica.txt'))
        else:
            np.testing.assert_allclose(batch['cameras'][0]['P'].data.numpy().ravel(), 
                np.loadtxt('./data/files/eval_cached_cameras_mp3d.txt'))
    except Exception as e:
        raise Exception("\n \nThere is an error with your setup or options. \
            \n\nYour results will NOT be comparable with results in the paper or online.")

METRICS = {
    "PSNR": lambda pred_imgs, key: psnr(
        pred_imgs["OutputImg"], pred_imgs[key]
    ).clamp(max=100),
    "PSNR_invis": lambda pred_imgs, key: psnr_mask(
        pred_imgs, key, True
    ).clamp(max=100),
    "PSNR_vis": lambda pred_imgs, key: psnr_mask(
        pred_imgs, key, False
    ).clamp(max=100),
    "SSIM": lambda pred_imgs, key: ssim_metric(
        pred_imgs["OutputImg"], pred_imgs[key]
    ),
    "SSIM_invis": lambda pred_imgs, key: ssim_mask(pred_imgs, key, True),
    "SSIM_vis": lambda pred_imgs, key: ssim_mask(pred_imgs, key, False),
    "PercSim": lambda pred_imgs, key: perceptual_sim(
        pred_imgs["OutputImg"], pred_imgs[key], vgg16
    ),
    "PercSim_invis": lambda pred_imgs, key: perceptual_sim_mask(
        pred_imgs, key, vgg16, True
    ),
    "PercSim_vis": lambda pred_imgs, key: perceptual_sim_mask(
        pred_imgs, key, vgg16, False
    ),
}

if __name__ == "__main__":
    print("STARTING MAIN METHOD...", flush=True)
    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    test_ops, _ = ArgumentParser().parse()

    # Load model to be tested
    MODEL_PATH = test_ops.old_model
    BATCH_SIZE = test_ops.batch_size

    opts = torch.load(MODEL_PATH)["opts"]
    print("Model is: ", MODEL_PATH)

    opts.image_type = test_ops.image_type
    opts.only_high_res = False

    opts.train_depth = False

    if test_ops.dataset:
        opts.dataset = test_ops.dataset

    Dataset = get_dataset(opts)
    model = get_model(opts)

    # Update parameters
    opts.render_ids = test_ops.render_ids
    opts.gpu_ids = test_ops.gpu_ids

    opts.images_before_reset = test_ops.images_before_reset

    torch_devices = [int(gpu_id.strip()) for gpu_id in opts.gpu_ids.split(",")]
    device = "cuda:" + str(torch_devices[0])

    if "sync" in opts.norm_G:
        model = convert_model(model)
        model = nn.DataParallel(model, torch_devices).to(device)
    else:
        model = nn.DataParallel(model, torch_devices).to(device)

    #  Load the original model to be tested
    model_to_test = BaseModel(model, opts)
    model_to_test.eval()
    model_to_test.load_state_dict(torch.load(MODEL_PATH)["state_dict"])

    # Load VGG16 for feature similarity
    vgg16 = PNet().to(device)
    vgg16.eval()

    # Create dummy depth model for doing sampling images
    sampled_model = Model(opts).to(device)
    sampled_model.eval()

    print("Loaded models...")

    model_to_test.eval()

    data = Dataset("test", opts, vectorize=False)
    dataloader = DataLoader(
        data,
        shuffle=False,
        drop_last=False,
        batch_size=BATCH_SIZE,
        num_workers=test_ops.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    iter_data_loader = iter(dataloader)
    next(iter_data_loader)

    N = test_ops.images_before_reset * 18 * BATCH_SIZE

    if not os.path.exists(test_ops.result_folder):
        os.makedirs(test_ops.result_folder)

    file_to_write = open(
        test_ops.result_folder + "/%s_results.txt" % test_ops.short_name, "w"
    )

    # Calculate the metrics and store for each index
    # and change in angle and translation
    results_all = {}
    for i in tqdm(range(0, N // BATCH_SIZE)):
        with torch.no_grad():
            _, pred_imgs, batch = model_to_test(
                iter_data_loader, isval=True, return_batch=True
            )

            _, new_imgs = sampled_model(batch)
            pred_imgs["SampledImg"] = new_imgs["SampledImg"] * 0.5 + 0.5

        # Check to make sure options were set right and this matches the setup
        # we used, so that numbers are comparable.
        if i == 0:
            check_initial_batch(batch, test_ops.dataset)

        # Obtain the angles and translation
        for batch_id in range(0, batch["cameras"][0]["P"].size(0)):
            dangles, dtrans = get_deltas(
                batch["cameras"][0]["P"][batch_id, 0:3, :],
                batch["cameras"][-1]["P"][batch_id, 0:3, :],
            )

        for metric, func in METRICS.items():
            key = "InputImg" if test_ops.test_input_image else "PredImg"
            t_results = func(pred_imgs, key)

            if not (metric in results_all.keys()):
                results_all[metric] = t_results.sum()
            else:
                results_all[metric] += t_results.sum()

            if i < 10:
                if not os.path.exists(test_ops.result_folder + "/%s" % metric):
                    os.makedirs(test_ops.result_folder + "/%s/" % metric)

                torchvision.utils.save_image(
                    pred_imgs["OutputImg"],
                    test_ops.result_folder
                    + "/%s/%03d_output_%s.png"
                    % (metric, i, test_ops.short_name),
                    pad_value=1,
                )
                
                torchvision.utils.save_image(
                    pred_imgs["InputImg"],
                    test_ops.result_folder
                    + "/%s/%03d_input_%s.png"
                    % (metric, i, test_ops.short_name),
                    pad_value=1,
                )

                if "SampledImg" in pred_imgs.keys():
                    torchvision.utils.save_image(
                        pred_imgs["SampledImg"],
                        test_ops.result_folder
                        + "/%s/%03d_sampled_%s.png"
                        % (metric, i, test_ops.short_name),
                        pad_value=1,
                    )
                    torchvision.utils.save_image(
                        (pred_imgs["SampledImg"] == pred_imgs["OutputImg"]).float(),
                        test_ops.result_folder
                        + "/%s/%03d_sampledmask_%s.png"
                        % (metric, i, test_ops.short_name),
                        pad_value=1,
                    )

                if "PredDepth" in pred_imgs.keys():
                    torchvision.utils.save_image(
                        pred_imgs["PredDepth"],
                        test_ops.result_folder
                        + "/%s/%03d_depth_%s.png"
                        % (metric, i, test_ops.short_name),
                        pad_value=1,
                        normalize=True,
                    )

                predimg = (
                    torchvision.utils.make_grid(
                        pred_imgs["PredImg"], pad_value=1
                    )
                    .clamp(min=0.00001, max=0.9999)
                    .permute(1, 2, 0)
                    .cpu()
                    .numpy()
                )
                predimg = cv2.cvtColor(
                    (predimg * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
                )
                for b in range(0, t_results.size(0)):
                    cv2.putText(
                        predimg,
                        "%0.4f" % t_results[b],
                        org=(258 * b + 10, 250),
                        fontFace=2,
                        fontScale=1,
                        color=(255, 255, 255),
                        bottomLeftOrigin=False,
                    )
                cv2.imwrite(
                    test_ops.result_folder
                    + "/%s/%03d_p_%s.png" % (metric, i, test_ops.short_name),
                    predimg,
                )

    for metric, result in results_all.items():
        file_to_write.write(
            "%s \t %0.5f \n" % (metric, result / float(BATCH_SIZE * (i + 1)))
        )

    file_to_write.close()

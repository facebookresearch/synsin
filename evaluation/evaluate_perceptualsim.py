# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import argparse
import glob
import os
from tqdm import tqdm

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from evaluation.metrics import perceptual_sim, psnr, ssim_metric
from models.networks.pretrained_networks import PNet

transform = transforms.Compose([transforms.ToTensor()])


def load_img(img_name, size=None):
    try:
        img = Image.open(img_name)

        if size:
            img = img.resize((size, size))

        img = transform(img).cuda()
        img = img.unsqueeze(0)
    except Exception as e:
        print("Failed at loading %s " % img_name)
        print(e)
        img = torch.zeros(1, 3, 256, 256).cuda()
    return img


def compute_perceptual_similarity(folder, pred_img, tgt_img, take_every_other):

    # Load VGG16 for feature similarity
    vgg16 = PNet().to("cuda")
    vgg16.eval()
    vgg16.cuda()

    values_percsim = []
    values_ssim = []
    values_psnr = []
    folders = os.listdir(folder)
    for i, f in tqdm(enumerate(sorted(folders))):
        pred_imgs = glob.glob(folder + f + "/" + pred_img)
        tgt_imgs = glob.glob(folder + f + "/" + tgt_img)
        assert len(tgt_imgs) == 1

        perc_sim = 10000
        ssim_sim = -10
        psnr_sim = -10
        for p_img in pred_imgs:
            t_img = load_img(tgt_imgs[0])
            p_img = load_img(p_img, size=t_img.size(2))
            t_perc_sim = perceptual_sim(p_img, t_img, vgg16).item()
            perc_sim = min(perc_sim, t_perc_sim)

            ssim_sim = max(ssim_sim, ssim_metric(p_img, t_img).item())
            psnr_sim = max(psnr_sim, psnr(p_img, t_img).item())

        values_percsim += [perc_sim]
        values_ssim += [ssim_sim]
        values_psnr += [psnr_sim]

    if take_every_other:
        n_valuespercsim = []
        n_valuesssim = []
        n_valuespsnr = []
        for i in range(0, len(values_percsim) // 2):
            n_valuespercsim += [
                min(values_percsim[2 * i], values_percsim[2 * i + 1])
            ]
            n_valuespsnr += [max(values_psnr[2 * i], values_psnr[2 * i + 1])]
            n_valuesssim += [max(values_ssim[2 * i], values_ssim[2 * i + 1])]

        values_percsim = n_valuespercsim
        values_ssim = n_valuesssim
        values_psnr = n_valuespsnr

    avg_percsim = np.mean(np.array(values_percsim))
    std_percsim = np.std(np.array(values_percsim))

    avg_psnr = np.mean(np.array(values_psnr))
    std_psnr = np.std(np.array(values_psnr))

    avg_ssim = np.mean(np.array(values_ssim))
    std_ssim = np.std(np.array(values_ssim))

    return {
        "Perceptual similarity": (avg_percsim, std_percsim),
        "PSNR": (avg_psnr, std_psnr),
        "SSIM": (avg_ssim, std_ssim),
    }


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--folder", type=str, default="")
    args.add_argument("--pred_image", type=str, default="")
    args.add_argument("--target_image", type=str, default="")
    args.add_argument("--take_every_other", action="store_true", default=False)
    args.add_argument("--output_file", type=str, default="")

    opts = args.parse_args()

    folder = opts.folder
    pred_img = opts.pred_image
    tgt_img = opts.target_image

    results = compute_perceptual_similarity(
        folder, pred_img, tgt_img, opts.take_every_other
    )

    f = open(opts.output_file, 'w')
    for key in results:
        print("%s for %s: \n" % (key, opts.folder))
        print(
            "\t {:0.4f} | {:0.4f} \n".format(results[key][0], results[key][1])
        )

        f.write("%s for %s: \n" % (key, opts.folder))
        f.write(
            "\t {:0.4f} | {:0.4f} \n".format(results[key][0], results[key][1])
        )

    f.close()

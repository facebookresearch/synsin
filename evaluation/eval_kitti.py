# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import os

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
from PIL import Image
from torch.utils.data import DataLoader

from models.base_model import BaseModel
from models.networks.sync_batchnorm import convert_model
from options.options import get_dataset, get_model
from options.test_options import ArgumentParser

torch.backends.cudnn.benchmark = True
torch.manual_seed(0)

class Dataset(data.Dataset):
    def __init__(self):
        self.path = os.environ["KITTI"]

        self.files = np.loadtxt('./data/files/kitti.txt', dtype=np.str)

        self.K = np.array(
                [718.9 / 256., 0., 128 / 256., 0, \
                 0., 718.9 / 256., 128 / 256., 0, \
                 0., 0., 1., 0., \
                 0., 0., 0., 1.]).reshape((4, 4)).astype(np.float32)
        self.invK = np.linalg.inv(self.K)

    def __len__(self):
        return len(self.files)

    def load_image(self, image_path):
        image = np.asarray(Image.open(image_path).convert('RGB'))
        image = image / 255. * 2. - 1.
        image = torch.from_numpy(image.astype(np.float32)).permute((2,0,1))
        return image

    def __getitem__(self, index):
        imgA = self.path + self.files[index,1] + '.png'
        imgB = self.path + self.files[index,0] + '.png'
        RT = self.files[index,2:].astype(np.float32)

        B = self.load_image(imgB) 
        A = self.load_image(imgA)

        RT = RT.astype(np.float32).reshape(4,4)
        RTinv = np.linalg.inv(RT).astype(np.float32)

        identity = torch.eye(4)

        offset = np.array(
            [[2, 0, -1, 0], [0, -2, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]],  # Flip ys to match habitat
            dtype=np.float32,
        )  # Make z negative to match habitat (which assumes a negative z)

        K = np.matmul(offset, self.K)

        Kinv = np.linalg.inv(K).astype(np.float32)

        return {'images' : [A, B], 'cameras' : [{'Pinv' : identity, 'P' : identity, 'K' : K, 'Kinv' : Kinv},
                                                {'Pinv' : RTinv, 'P' : RT, 'K' : K, 'Kinv' : Kinv}]
        }


if __name__ == "__main__":
    test_ops, _ = ArgumentParser().parse()

    # Load model to be tested
    MODEL_PATH = test_ops.old_model
    BATCH_SIZE = test_ops.batch_size

    opts = torch.load(MODEL_PATH)["opts"]

    model = get_model(opts)

    opts.render_ids = test_ops.render_ids
    opts.gpu_ids = test_ops.gpu_ids

    torch_devices = [int(gpu_id.strip()) for gpu_id in opts.gpu_ids.split(",")]
    print(torch_devices)
    device = "cuda:" + str(torch_devices[0])

    if "sync" in opts.norm_G:
        model = convert_model(model)
        model = nn.DataParallel(model, torch_devices).to(device)
    else:
        model = nn.DataParallel(model, torch_devices).to(device)

    #  Load the original model to be tested
    model_to_test = BaseModel(model, opts)
    model_to_test.eval()

    # Allow for different image sizes
    state_dict = model_to_test.state_dict()
    pretrained_dict = {
        k: v
        for k, v in torch.load(MODEL_PATH)["state_dict"].items()
        if not ("xyzs" in k) and not ("ones" in k)
    }
    state_dict.update(pretrained_dict)

    model_to_test.load_state_dict(state_dict)

    print(opts)
    # Update parameters
    opts.render_ids = test_ops.render_ids
    opts.gpu_ids = test_ops.gpu_ids


    print("Loaded models...")

    # Load the dataset which is the set of images that came
    # from running the baselines' result scripts
    data = Dataset()

    model_to_test.eval()

    # Iterate through the dataset, predicting new views
    data_loader = DataLoader(data, batch_size=1, shuffle=False)
    iter_data_loader = iter(data_loader)

    for i in range(0, len(data_loader)):
        print(i, len(data_loader), flush=True)
        _, pred_imgs, batch = model_to_test(
            iter_data_loader, isval=True, return_batch=True
        )

        if not os.path.exists(
            test_ops.result_folder
            + "/%d/" % (i)
        ):
            os.makedirs(
                test_ops.result_folder
                + "/%d/" % (i)
            )

        torchvision.utils.save_image(
            pred_imgs["PredImg"],
            test_ops.result_folder
            + "/%d/im_res.png" % (i),
        )
        torchvision.utils.save_image(
            pred_imgs["OutputImg"],
            test_ops.result_folder
            + "/%d/im_B.png" % (i),
        )
        torchvision.utils.save_image(
            pred_imgs["InputImg"],
            test_ops.result_folder
            + "/%d/im_A.png" % (i),
        )


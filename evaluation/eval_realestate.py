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
    def __init__(self, W=256):

        self.base_path = os.environ['REALESTATE']
        self.files = np.loadtxt('./data/files/realestate.txt', dtype=np.str)

        self.offset = np.array(
            [[2, 0, -1], [0, -2, 1], [0, 0, -1]], dtype=np.float32
        )

        self.K = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        self.invK = np.linalg.inv(self.K)

        self.input_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((W, W)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                ),
            ]
        )

        self.W = W

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        # Then load the image and generate that
        file_name = self.files[index]

        src_image_name = (
            self.base_path
            + '/%s/%s.png' % (file_name[0], file_name[1])
        )
        tgt_image_name = (
            self.base_path
            + '/%s/%s.png' % (file_name[0], file_name[2])
        )

        intrinsics = file_name[3:7].astype(np.float32) / float(self.W)
        src_pose = file_name[7:19].astype(np.float32).reshape(3, 4)
        tgt_pose = file_name[19:].astype(np.float32).reshape(3, 4)

        src_image = self.input_transform(Image.open(src_image_name))
        tgt_image = self.input_transform(Image.open(tgt_image_name))

        poses = [src_pose, tgt_pose]
        cameras = []

        for pose in poses:

            origK = np.array(
                [
                    [intrinsics[0], 0, intrinsics[2]],
                    [0, intrinsics[1], intrinsics[3]],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )
            K = np.matmul(self.offset, origK)

            P = pose
            P = np.matmul(K, P)
            # Merge these together to match habitat
            P = np.vstack((P, np.zeros((1, 4)))).astype(np.float32)
            P[3, 3] = 1

            # Now artificially flip x/ys to match habitat
            Pinv = np.linalg.inv(P)

            cameras += [{"P": P, "Pinv": Pinv, "K": self.K, "Kinv": self.invK}]

        return {"images": [src_image, tgt_image], "cameras": cameras}


if __name__ == "__main__":
    test_ops, _ = ArgumentParser().parse()

    # Load model to be tested
    MODEL_PATH = test_ops.old_model
    BATCH_SIZE = test_ops.batch_size

    opts = torch.load(MODEL_PATH)["opts"]
    opts.isTrain = True
    opts.only_high_res = False
    opts.lr_d = 0.001

    opts.train_depth = False
    print(opts)

    DatasetTrain = get_dataset(opts)
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
    data = Dataset(W=opts.W)

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
            + "/%04d/" % (i)
        ):
            os.makedirs(
                test_ops.result_folder
                + "/%04d/" % (i)
            )

        torchvision.utils.save_image(
            pred_imgs["PredImg"],
            test_ops.result_folder
            + "/%04d/output_image_.png" % (i),
        )
        torchvision.utils.save_image(
            pred_imgs["OutputImg"],
            test_ops.result_folder
            + "/%04d/tgt_image_.png" % (i),
        )
        torchvision.utils.save_image(
            pred_imgs["InputImg"],
            test_ops.result_folder
            + "/%04d/input_image_.png" % (i),
        )

        print(
            pred_imgs["PredImg"].mean().item(),
            pred_imgs["PredImg"].std().item(),
            flush=True,
        )

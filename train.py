# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import os
import signal
import time

import torch
import torch.nn as nn
import torchvision
from tensorboardX import SummaryWriter as tensorboardWriter
from torch.multiprocessing import set_start_method
from torch.utils.data import DataLoader

from models.base_model import BaseModel
from models.networks.sync_batchnorm import convert_model
from options.options import get_dataset, get_model
from options.train_options import (
    ArgumentParser,
    get_log_path,
    get_model_path,
    get_timestamp,
)

torch.backends.cudnn.benchmark = True

SIGNAL_RECEIVED = False

if os.environ['USE_SLURM'] == 1:
    def SIGTERMHandler(a, b):
        print("received sigterm")
        pass


    def signalHandler(a, b):
        global SIGNAL_RECEIVED, args
        print("Signal received", a, time.time(), flush=True)
        SIGNAL_RECEIVED = True

        """ If HALT file exists, which means the job is done, exit peacefully.
        """
        if os.path.isfile(HALT_filename):
            print("Job is done, exiting")
            exit(0)

        return


    def trigger_job_requeue():
        """ Submit a new job to resume from checkpoint.
        """
        print("Starting to trigger job requeue...", flush=True)
        print(os.environ["SLURM_PROCID"], flush=True)
        print(os.getpid())
        print(MAIN_PID)
        print(os.environ["SLURM_PROCID"] == "0")
        print(os.getpid() == MAIN_PID)
        if os.environ["SLURM_PROCID"] == "0" and os.getpid() == MAIN_PID:
            """ BE AWARE OF subprocesses that your program spawns.
            Only the main process on slurm procID = 0 resubmits the job.
            In pytorch imagenet example, by default it spawns 4
            (specified in -j) subprocesses for data loading process,
            both parent process and child processes will receive the signal.
            Please only submit the job in main process,
            otherwise the job queue will be filled up exponentially.
            Command below can be used to check the pid of running processes.
            print('pid: ', os.getpid(), ' ppid: ', os.getppid(), flush=True)
            """
            print("time is up, back to slurm queue", flush=True)
            command = "scontrol requeue " + os.environ["SLURM_JOB_ID"]
            print(command)
            if os.system(command):
                raise RuntimeError("requeue failed")
            print("New job submitted to the queue", flush=True)
        exit(0)


    # ''' Install signal handler
    # '''
    MAIN_PID = os.getpid()
    signal.signal(signal.SIGUSR1, signalHandler)
    signal.signal(signal.SIGTERM, SIGTERMHandler)
    print("Signal handler installed", flush=True)


def train(epoch, data_loader, model, log_path, plotter, opts):
    print("At train", flush=True)

    losses = {}
    iter_data_loader = iter(data_loader)

    for iteration in range(1, min(501, len(data_loader))):
        t_losses, output_image = model(
            iter_data_loader, isval=False, num_steps=opts.num_accumulations
        )

        for l in t_losses.keys():
            if l in losses.keys():
                losses[l] = t_losses[l].cpu().mean().detach().item() + losses[l]
            else:
                losses[l] = t_losses[l].cpu().mean().detach().item()
        if iteration % 250 == 0 or iteration == 1:
            for add_im in output_image.keys():
                if iteration == 1 and os.environ['DEBUG']:
                    torchvision.utils.save_image(
                        output_image[add_im][0:8, :, :, :].cpu().data,
                        "./debug/Image_train/%d_%s.png" % (iteration, add_im),
                        normalize=("Depth" in add_im),
                    )
                
                plotter.add_image(
                    "Image_train/%d_%s" % (iteration, add_im),
                    torchvision.utils.make_grid(
                        output_image[add_im][0:8, :, :, :].cpu().data,
                        normalize=("Depth" in add_im),
                    ),
                    epoch,
                )

        if iteration % 1 == 0:
            str_to_print = "Train: Epoch {}: {}/{} with ".format(
                epoch, iteration, len(data_loader)
            )
            for l in losses.keys():
                str_to_print += " %s : %0.4f | " % (
                    l,
                    losses[l] / float(iteration),
                )
            print(str_to_print, flush=True)

        if SIGNAL_RECEIVED:
            # checkpoint(model, opts.model_epoch_path, CHECKPOINT_tempfile)
            trigger_job_requeue()
            raise SystemExit

        for l in t_losses.keys():
            plotter.add_scalars(
                "%s_iter" % l,
                {"train": t_losses[l].cpu().mean().detach().item()},
                epoch * 500 + iteration,
            )

    return {l: losses[l] / float(iteration) for l in losses.keys()}


def val(epoch, data_loader, model, log_path, plotter):

    losses = {}

    iter_data_loader = iter(data_loader)
    for iteration in range(1, min(51, len(data_loader))):
        t_losses, output_image = model(
            iter_data_loader, isval=True, num_steps=1
        )
        for l in t_losses.keys():
            if l in losses.keys():
                losses[l] = t_losses[l].cpu().mean().item() + losses[l]
            else:
                losses[l] = t_losses[l].cpu().mean().item()
        if iteration % 100 == 0 or iteration == 1:
            for add_im in output_image.keys():
                
                plotter.add_image(
                    "Image_val/%d_%s" % (iteration, add_im),
                    torchvision.utils.make_grid(
                        output_image[add_im][0:8, :, :, :].cpu().data,
                        normalize=("Depth" in add_im),
                    ),
                    epoch,
                )

        if SIGNAL_RECEIVED:
            trigger_job_requeue()
            raise SystemExit

        if iteration % 1 == 0:
            str_to_print = "Val: Epoch {}: {}/{} with ".format(
                epoch, iteration, len(data_loader)
            )
            for l in losses.keys():
                str_to_print += " %s : %0.4f | " % (
                    l,
                    losses[l] / float(iteration),
                )
            print(str_to_print, flush=True)

        for l in t_losses.keys():
            plotter.add_scalars(
                "%s_iter" % l,
                {"val": t_losses[l].cpu().mean().detach().item()},
                epoch * 50 + iteration,
            )

    return {l: losses[l] / float(iteration) for l in losses.keys()}


def checkpoint(model, save_path, CHECKPOINT_tempfile):
    if model.use_discriminator:
        checkpoint_state = {
            "state_dict": model.state_dict(),
            "optimizerG": model.optimizer_G.state_dict(),
            "epoch": model.epoch,
            "optimizerD": model.optimizer_D.state_dict(),
            "opts": opts,
        }

    else:
        checkpoint_state = {
            "state_dict": model.state_dict(),
            "optimizerG": model.optimizer_G.state_dict(),
            "epoch": model.epoch,
            "opts": opts,
        }

    torch.save(checkpoint_state, CHECKPOINT_tempfile)
    if os.path.isfile(CHECKPOINT_tempfile):
        os.rename(CHECKPOINT_tempfile, save_path)


def run(model, Dataset, log_path, plotter, CHECKPOINT_tempfile):
    print("Starting run...", flush=True)

    opts.best_epoch = 0
    opts.best_loss = -1000
    if os.path.exists(opts.model_epoch_path) and opts.resume:
        past_state = torch.load(opts.model_epoch_path)
        print("Continuing epoch ... %d" % (past_state['opts'].continue_epoch + 1), flush=True)
        model.load_state_dict(torch.load(opts.model_epoch_path)["state_dict"])
        model.optimizer_D.load_state_dict(
            torch.load(opts.model_epoch_path)["optimizerD"]
        )
        model.optimizer_G.load_state_dict(
            torch.load(opts.model_epoch_path)["optimizerG"]
        )

        opts.continue_epoch = past_state["opts"].continue_epoch + 1
        opts.current_episode_train = past_state["opts"].current_episode_train
        opts.current_episode_val = past_state["opts"].current_episode_val
        opts.best_epoch = past_state["opts"].best_epoch
        opts.best_loss = past_state["opts"].best_loss
    elif opts.resume:
        print("WARNING: Model path does not exist?? ")
        print(opts.model_epoch_path)

    print("Loading train dataset ....", flush=True)
    train_set = Dataset("train", opts)

    train_data_loader = DataLoader(
        dataset=train_set,
        num_workers=opts.num_workers,
        batch_size=opts.batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
    )

    print("Loaded train dataset ...", flush=True)

    for epoch in range(opts.continue_epoch, opts.max_epoch):
        print("Starting epoch %d" % epoch, flush=True)
        opts.continue_epoch = epoch
        model.epoch = epoch
        model.train()

        train_loss = train(
            epoch, train_data_loader, model, log_path, plotter, opts
        )

        model.eval()
        with torch.no_grad():

            model.eval()
            train_set.toval(
                epoch=0
            )  # Hack because don't want to keep reloading the environments
            loss = val(epoch, train_data_loader, model, log_path, plotter)
            train_set.totrain(epoch=epoch + 1 + opts.seed)

        for l in train_loss.keys():
            if l in loss.keys():
                plotter.add_scalars(
                    "%s_epoch" % l,
                    {"train": train_loss[l], "val": loss[l]},
                    epoch,
                )
            else:
                plotter.add_scalars(
                    "%s_epoch" % l, {"train": train_loss[l]}, epoch
                )

        if loss["psnr"] > opts.best_loss:
            checkpoint(
                model, opts.model_epoch_path + "best", CHECKPOINT_tempfile
            )
            opts.best_epoch = epoch
            opts.best_loss = loss["psnr"]

        checkpoint(model, opts.model_epoch_path, CHECKPOINT_tempfile)

        if epoch % 50 == 0:
            checkpoint(
                model,
                opts.model_epoch_path + "ep%d" % epoch,
                CHECKPOINT_tempfile,
            )

    if epoch == 500 - 1:
        open(HALT_filename, "a").close()


if __name__ == "__main__":
    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    opts, _ = ArgumentParser().parse()

    opts.isTrain = True
    timestamp = get_timestamp()
    print("Timestamp ", timestamp, flush=True)

    opts.model_epoch_path = get_model_path(timestamp, opts)
    print("Model ", opts.model_epoch_path, flush=True)

    Dataset = get_dataset(opts)
    model = get_model(opts)

    log_path = get_log_path(timestamp, opts)
    print(log_path)
    plotter = tensorboardWriter(logdir=log_path % "tensorboard")

    torch_devices = [int(gpu_id.strip()) for gpu_id in opts.gpu_ids.split(",")]
    print(torch_devices)
    device = "cuda:" + str(torch_devices[0])

    if "sync" in opts.norm_G:
        model = convert_model(model)
        model = nn.DataParallel(model, torch_devices).to(device)
    else:
        model = nn.DataParallel(model, torch_devices).to(device)

    CHECKPOINT_tempfile = opts.model_epoch_path + ".tmp"
    global CHECKPOINT_filename
    CHECKPOINT_filename = CHECKPOINT_tempfile
    HALT_filename = "HALT"

    if os.path.isfile(CHECKPOINT_tempfile):
        os.remove(CHECKPOINT_tempfile)

    if opts.load_old_model:
        # opt = torch.load(opts.old_model)['opts']
        model = BaseModel(model, opts)

        # Allow for different image sizes
        state_dict = model.state_dict()
        pretrained_dict = {
            k: v
            for k, v in torch.load(opts.old_model)["state_dict"].items()
            if not ("xyzs" in k) and not ("ones" in k)
        }
        state_dict.update(pretrained_dict)

        model.load_state_dict(state_dict)

        run(model, Dataset, log_path, plotter, CHECKPOINT_tempfile)

    else:
        model = BaseModel(model, opts)

        run(model, Dataset, log_path, plotter, CHECKPOINT_tempfile)

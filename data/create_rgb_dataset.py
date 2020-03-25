# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# Taken from https://github.com/facebookresearch/splitnet

import gzip
import os
from typing import Dict

import habitat
import habitat.datasets.pointnav.pointnav_dataset as mp3d_dataset
import numpy as np
import quaternion
import torch
import torchvision.transforms as transforms
import tqdm
from habitat.config.default import get_config
from habitat.datasets import make_dataset

from geometry.camera_transformations import get_camera_matrices
from utils.jitter import jitter_quaternions


def _load_datasets(config_keys, dataset, data_path, scenes_path, num_workers):
    # For each scene, create a new dataset which is added with the config
    # to the vector environment.

    print(len(dataset.episodes))
    datasets = []
    configs = []

    num_episodes_per_worker = len(dataset.episodes) / float(num_workers)

    for i in range(0, min(len(dataset.episodes), num_workers)):
        config = make_config(*config_keys)
        config.defrost()

        dataset_new = mp3d_dataset.PointNavDatasetV1()
        with gzip.open(data_path, "rt") as f:
            dataset_new.from_json(f.read())
            dataset_new.episodes = dataset_new.episodes[
                int(i * num_episodes_per_worker) : int(
                    (i + 1) * num_episodes_per_worker
                )
            ]

            for episode_id in range(0, len(dataset_new.episodes)):
                dataset_new.episodes[episode_id].scene_id = \
                    dataset_new.episodes[episode_id].scene_id.replace(
                        '/checkpoint/erikwijmans/data/mp3d/',
                        scenes_path)

        config.SIMULATOR.SCENE = str(dataset_new.episodes[0].scene_id)
        config.freeze()

        datasets += [dataset_new]
        configs += [config]
    return configs, datasets


def make_config(
    config, gpu_id, split, data_path, sensors, resolution, scenes_dir
):
    config = get_config(config)
    config.defrost()
    config.TASK.NAME = "Nav-v0"
    config.TASK.MEASUREMENTS = []
    config.DATASET.SPLIT = split
    config.DATASET.POINTNAVV1.DATA_PATH = data_path
    config.DATASET.SCENES_DIR = scenes_dir
    config.HEIGHT = resolution
    config.WIDTH = resolution
    for sensor in sensors:
        config.SIMULATOR[sensor]["HEIGHT"] = resolution
        config.SIMULATOR[sensor]["WIDTH"] = resolution

    config.TASK.HEIGHT = resolution
    config.TASK.WIDTH = resolution
    config.SIMULATOR.TURN_ANGLE = 15
    config.SIMULATOR.FORWARD_STEP_SIZE = 0.1  # in metres
    config.SIMULATOR.AGENT_0.SENSORS = sensors
    config.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False

    config.SIMULATOR.DEPTH_SENSOR.HFOV = 90

    config.ENVIRONMENT.MAX_EPISODE_STEPS = 2 ** 32
    config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpu_id
    return config


class RandomImageGenerator(object):
    def __init__(self, split, gpu_id, opts, vectorize=False, seed=0) -> None:
        self.vectorize = vectorize

        print("gpu_id", gpu_id)
        resolution = opts.W
        if opts.use_semantics:
            sensors = ["RGB_SENSOR", "DEPTH_SENSOR", "SEMANTIC_SENSOR"]
        else:
            sensors = ["RGB_SENSOR", "DEPTH_SENSOR"]
        if split == "train":
            data_path = opts.train_data_path
        elif split == "val":
            data_path = opts.val_data_path
        elif split == "test":
            data_path = opts.test_data_path
        else:
            raise Exception("Invalid split")
        unique_dataset_name = opts.dataset

        self.num_parallel_envs = 5

        self.images_before_reset = opts.images_before_reset
        config = make_config(
            opts.config,
            gpu_id,
            split,
            data_path,
            sensors,
            resolution,
            opts.scenes_dir,
        )
        data_dir = os.path.join(
            "data/scene_episodes/", unique_dataset_name + "_" + split
        )
        self.dataset_name = config.DATASET.TYPE
        print(data_dir)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        data_path = os.path.join(data_dir, "dataset_one_ep_per_scene.json.gz")
        # Creates a dataset where each episode is a random spawn point in each scene.
        print("One ep per scene", flush=True)
        if not (os.path.exists(data_path)):
            print("Creating dataset...", flush=True)
            dataset = make_dataset(config.DATASET.TYPE, config=config.DATASET)
            # Get one episode per scene in dataset
            scene_episodes = {}
            for episode in tqdm.tqdm(dataset.episodes):
                if episode.scene_id not in scene_episodes:
                    scene_episodes[episode.scene_id] = episode

            scene_episodes = list(scene_episodes.values())
            dataset.episodes = scene_episodes
            if not os.path.exists(data_path):
                # Multiproc do check again before write.
                json = dataset.to_json().encode("utf-8")
                with gzip.GzipFile(data_path, "w") as fout:
                    fout.write(json)
            print("Finished dataset...", flush=True)

        # Load in data and update the location to the proper location (else
        # get a weird, uninformative, error -- Affine2Dtransform())
        dataset = mp3d_dataset.PointNavDatasetV1()
        with gzip.open(data_path, "rt") as f:
            dataset.from_json(f.read())

            for i in range(0, len(dataset.episodes)):
                dataset.episodes[i].scene_id = dataset.episodes[i].scene_id.replace(
                    '/checkpoint/erikwijmans/data/mp3d/',
                        opts.scenes_dir + '/mp3d/')

        config.TASK.SENSORS = ["POINTGOAL_SENSOR"]

        config.freeze()

        self.rng = np.random.RandomState(seed)

        # Now look at vector environments
        if self.vectorize:
            configs, datasets = _load_datasets(
                (
                    opts.config,
                    gpu_id,
                    split,
                    data_path,
                    sensors,
                    resolution,
                    opts.scenes_dir,
                ),
                dataset,
                data_path,
                opts.scenes_dir + '/mp3d/',
                num_workers=self.num_parallel_envs,
            )
            num_envs = len(configs)

            env_fn_args = tuple(zip(configs, datasets, range(num_envs)))
            envs = habitat.VectorEnv(
                env_fn_args=env_fn_args,
                multiprocessing_start_method="forkserver",
            )

            self.env = envs
            self.num_train_envs = int(0.9 * (self.num_parallel_envs))
            self.num_val_envs = self.num_parallel_envs - self.num_train_envs
        else:
            self.env = habitat.Env(config=config, dataset=dataset)
            self.env_sim = self.env.sim
            self.rng.shuffle(self.env.episodes)
            self.env_sim = self.env.sim

        self.num_samples = 0

        # Set up intrinsic parameters
        self.hfov = config.SIMULATOR.DEPTH_SENSOR.HFOV * np.pi / 180.0
        self.W = resolution
        self.K = np.array(
            [
                [1.0 / np.tan(self.hfov / 2.0), 0.0, 0.0, 0.0],
                [0, 1.0 / np.tan(self.hfov / 2.0), 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        self.invK = np.linalg.inv(self.K)

        self.config = config
        self.opts = opts

        if self.opts.normalize_image:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )  # Using same normalization as BigGan
        else:
            self.transform = transforms.ToTensor()

    def get_vector_sample(self, index, num_views, isTrain=True):
        if self.num_samples % self.images_before_reset == 0:
            self.env.reset()

        # Randomly choose an index of given environments
        if isTrain:
            index = index % self.num_train_envs
        else:
            index = (index % self.num_val_envs) + self.num_train_envs

        depths = []
        rgbs = []
        cameras = []
        semantics = []

        orig_location = np.array(self.env.sample_navigable_point(index))
        rand_angle = self.rng.uniform(0, 2 * np.pi)

        orig_rotation = [0, np.sin(rand_angle / 2), 0, np.cos(rand_angle / 2)]
        obs = self.env.get_observations_at(
            index, position=orig_location, rotation=orig_rotation
        )
        for i in range(0, num_views):
            rand_location = orig_location.copy()
            rand_rotation = orig_rotation.copy()
            if self.opts.image_type == "translation":
                rand_location[[0]] = (
                    orig_location[[0]] + self.rng.rand() * 0.2 - 0.1
                )
            elif self.opts.image_type == "outpaint":
                rand_rotation = jitter_quaternions(
                    quaternion.from_float_array(orig_rotation),
                    self.rng,
                    angle=10,
                )
                rand_rotation = quaternion.as_float_array(
                    rand_rotation
                ).tolist()
            elif self.opts.image_type == "fixedRT_baseline":
                rand_location = self.rand_location
                rotation = self.rand_rotation
            else:
                rand_location[0] = (
                    orig_location[0] + self.rng.rand() * 0.32 - 0.15
                )
                rand_rotation = jitter_quaternions(
                    quaternion.from_float_array(orig_rotation),
                    self.rng,
                    angle=10,
                )
                rand_rotation = quaternion.as_float_array(
                    rand_rotation
                ).tolist()

            obs = self.env.get_observations_at(
                index, position=rand_location, rotation=rand_rotation
            )

            depths += [torch.Tensor(obs["depth"][..., 0]).unsqueeze(0)]
            rgbs += [self.transform(obs["rgb"].astype(np.float32) / 256.0)]

            if "semantic" in obs.keys():
                instance_semantic = torch.Tensor(
                    obs["semantic"].astype(np.int32)
                ).unsqueeze(0)
                class_semantic = torch.zeros(instance_semantic.size()).long()

                id_to_label = {
                    int(obj.id.split("_")[-1]): obj.category.index()
                    for obj in self.env.sim.semantic_annotations().objects
                }

                for id_scene in id_to_label.keys():
                    class_semantic[instance_semantic == id_scene] = id_to_label[
                        id_scene
                    ]

                semantics += [class_semantic]

            position, rotation = self.env.get_agent_state(index)
            rotation = quaternion.as_rotation_matrix(
                quaternion.from_float_array(rotation)
            )
            P, Pinv = get_camera_matrices(position=position, rotation=rotation)
            cameras += [{"P": P, "Pinv": Pinv, "K": self.K, "Kinv": self.invK}]

        self.num_samples += 1
        if len(semantics) > 0:
            return {
                "images": rgbs,
                "depths": depths,
                "cameras": cameras,
                "semantics": semantics,
            }

        return {"images": rgbs, "depths": depths, "cameras": cameras}

    def get_singleenv_sample(self, num_views) -> Dict[str, np.ndarray]:

        if self.num_samples % self.images_before_reset == 0:
            old_env = self.env._current_episode_index
            self.env.reset()
            print(
                "RESETTING %d to %d \n"
                % (old_env, self.env._current_episode_index),
                flush=True,
            )

        depths = []
        rgbs = []
        cameras = []
        semantics = []

        rand_location = self.env_sim.sample_navigable_point()
        if self.opts.image_type == "fixedRT_baseline":
            rand_angle = self.angle_rng.uniform(0, 2 * np.pi)
        else:
            rand_angle = self.rng.uniform(0, 2 * np.pi)
        rand_rotation = [0, np.sin(rand_angle / 2), 0, np.cos(rand_angle / 2)]
        obs = self.env_sim.get_observations_at(
            position=rand_location,
            rotation=rand_rotation,
            keep_agent_at_new_pose=True,
        )

        for i in range(0, num_views):
            position = rand_location.copy()
            rotation = rand_rotation.copy()
            if self.opts.image_type == "translation":
                position[0] = position[0] + self.rng.rand() * 0.2 - 0.1
            elif self.opts.image_type == "outpaint":
                rotation = quaternion.as_float_array(
                    jitter_quaternions(
                        quaternion.from_float_array(rand_rotation),
                        self.rng,
                        angle=10,
                    )
                ).tolist()
            elif self.opts.image_type == "fixedRT_baseline":
                rand_location = self.rand_location
                rotation = self.rand_rotation

            else:
                position[0] = position[0] + self.rng.rand() * 0.3 - 0.15
                rotation = quaternion.as_float_array(
                    jitter_quaternions(
                        quaternion.from_float_array(rand_rotation),
                        self.rng,
                        angle=10,
                    )
                ).tolist()

            obs = self.env_sim.get_observations_at(
                position=position,
                rotation=rotation,
                keep_agent_at_new_pose=True,
            )

            depths += [torch.Tensor(obs["depth"][..., 0]).unsqueeze(0)]
            rgbs += [self.transform(obs["rgb"].astype(np.float32) / 256.0)]

            if "semantic" in obs.keys():
                instance_semantic = torch.Tensor(
                    obs["semantic"].astype(np.int32)
                ).unsqueeze(0)
                class_semantic = torch.zeros(instance_semantic.size()).long()

                id_to_label = {
                    int(obj.id.split("_")[-1]): obj.category.index()
                    for obj in self.env.sim.semantic_annotations().objects
                }

                for id_scene in np.unique(instance_semantic.numpy()):
                    class_semantic[instance_semantic == id_scene] = id_to_label[
                        id_scene
                    ]

                semantics += [class_semantic]

            agent_state = self.env_sim.get_agent_state().sensor_states["depth"]
            rotation = quaternion.as_rotation_matrix(agent_state.rotation)
            position = agent_state.position
            P, Pinv = get_camera_matrices(position=position, rotation=rotation)
            cameras += [{"P": P, "Pinv": Pinv, "K": self.K, "Kinv": self.invK}]

        self.num_samples += 1
        if len(semantics) > 0:
            return {
                "images": rgbs,
                "depths": depths,
                "cameras": cameras,
                "semantics": semantics,
            }

        return {"images": rgbs, "depths": depths, "cameras": cameras}

    def get_sample(self, index, num_views, isTrain):
        if self.vectorize:
            return self.get_vector_sample(index, num_views, isTrain)
        else:
            return self.get_singleenv_sample(num_views)

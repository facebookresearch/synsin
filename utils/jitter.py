# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import numpy as np
import quaternion

def jitter_quaternions(original_quaternion, rnd, angle=30.0):
    original_euler = quaternion.as_euler_angles(original_quaternion)
    euler_angles = np.array(
        [
            (rnd.rand() - 0.5) * np.pi * angle / 180.0 + original_euler[0],
            (rnd.rand() - 0.5) * np.pi * angle / 180.0 + original_euler[1],
            (rnd.rand() - 0.5) * np.pi * angle / 180.0 + original_euler[2],
        ]
    )
    quaternions = quaternion.from_euler_angles(euler_angles)

    return quaternions

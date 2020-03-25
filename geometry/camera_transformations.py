# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import numpy as np
import torch


def invert_RT(RT):
    """ Given an RT matrix (e.g. [R | T]) matrix where R is
    indeed valid, then inverts this.
    """
    R = RT[:, 0:3, 0:3]
    T = RT[:, :, 3:]

    # Get inverse of the rotation matrix
    Rinv = R.permute(0, 2, 1)
    Tinv = -Rinv.bmm(T)

    RTinv = torch.cat((Rinv, Tinv), 2)

    return RTinv


def invert_K(K):
    """ Given a K matrix (an intrinsic matrix) of the form
    [f 0 px]
    [0 f py]
    [0 0  1], inverts it.
    """
    K_inv = (
        torch.eye(K.size(1)).to(K.device).unsqueeze(0).repeat(K.size(0), 1, 1)
    )

    K_inv[:, 0, 0] = 1 / K[:, 0, 0]
    K_inv[:, 0, 2] = -K[:, 0, 2] / K[:, 0, 0]
    K_inv[:, 1, 1] = 1 / K[:, 1, 1]
    K_inv[:, 1, 2] = -K[:, 1, 2] / K[:, 1, 1]

    return K_inv


def get_camera_matrices(position, rotation):

    Pinv = np.eye(4)
    Pinv[0:3, 0:3] = rotation
    Pinv[0:3, 3] = position

    P = np.linalg.inv(Pinv)

    return P.astype(np.float32), Pinv.astype(np.float32)


if __name__ == "__main__":
    # Test the inversion code
    # 1. Test the RT

    rotate = np.linalg.qr(np.random.randn(3, 3))[0]
    R = torch.Tensor(rotate)
    translation = np.random.randn(3, 1)
    T = torch.Tensor(translation)

    RT = torch.cat((R, T), 1).unsqueeze(0)

    RTinv = invert_RT(RT)
    print(RT[:, 0:3, 0:3].bmm(RTinv[:, 0:3, 0:3]))
    x = torch.randn(1, 4, 1)
    x[0, 3, 0] = 1
    xp = RT.bmm(x)
    xp = torch.cat((xp, torch.ones((1, 1, 1))), 1)
    print(RTinv.bmm(xp) - x[:, 0:3, :])

    K = torch.eye(3).unsqueeze(0).repeat(2, 1, 1)
    K[0, 0, 0] = torch.randn(1)
    K[0, 1, 1] = torch.randn(1)
    K[0, 0, 2] = torch.randn(1)
    K[1, 1, 2] = torch.randn(1)
    K[1, 0, 0] = torch.randn(1)
    K[1, 1, 1] = torch.randn(1)
    K[1, 0, 2] = torch.randn(1)
    K[1, 1, 2] = torch.randn(1)

    Kinv = invert_K(K)
    print(Kinv.bmm(K))

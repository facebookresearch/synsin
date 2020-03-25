# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from math import sqrt

import numpy as np


def get_deltas(mat1, mat2):
    mat1 = np.vstack((mat1, np.array([0, 0, 0, 1])))
    mat2 = np.vstack((mat2, np.array([0, 0, 0, 1])))

    dMat = np.matmul(np.linalg.inv(mat1), mat2)
    dtrans = dMat[0:3, 3] ** 2
    dtrans = sqrt(dtrans.sum())

    origVec = np.array([[0], [0], [1]])
    rotVec = np.matmul(dMat[0:3, 0:3], origVec)
    arccos = (rotVec * origVec).sum() / sqrt((rotVec ** 2).sum())
    dAngle = np.arccos(arccos) * 180.0 / np.pi

    return dAngle, dtrans

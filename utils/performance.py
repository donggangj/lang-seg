import time
from typing import Callable

import numpy as np
from numpy import ndarray


def iterate_time(func: Callable, *x_in, n_repeat=10):
    outputs = []
    ts = []
    t0 = time.time()
    for _ in range(n_repeat):
        output = func(*x_in)
        outputs.append(output)
        t1 = time.time()
        ts.append((t1 - t0) * 1000)
        t0 = t1
    return ts, outputs


def calc_loss(pred: ndarray, ref: ndarray):
    ae_mat = abs(pred[0] - ref)
    mae = ae_mat.mean()
    se_mat = ae_mat ** 2
    rmse = np.sqrt(se_mat.mean())
    return mae, rmse

__version__ = '3.1'
__author__ = 'Jannick Stranghöner'

from copy import deepcopy
from typing import TypeVar, Generic, Tuple, List, Union, Optional, Callable, Generator
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from constraints import CIEQC, CEQC, DIEQC, DEQC
from learner import BaseLearner
from regions import SamplingRegion

Shape = TypeVar("Shape")
DType = TypeVar("DType")


class Array(np.ndarray, Generic[Shape, DType]):
    """
    Use this to type-annotate numpy arrays, e.g.

        def transform_image(image: Array['H,W,3', np.uint8], ...):
            ...

    """
    pass


def CEQC_to_CIEQCs(con: CEQC | DEQC,
                   eps: float = 1e-3) -> CIEQC | CEQC:
    """Turns equality constraint into inequality constraint with tighter bounds."""
    if callable(con.Value):
        ub = lambda x: con.Value(x) + eps
        lb = lambda x: con.Value(x) - eps
    else:
        ub = con.Value + eps
        lb = con.Value - eps
    if isinstance(con, DEQC):
        return DIEQC(u=con.u, partial=con.partial, factors=con.factors,
                     maxValue=ub, minValue=lb)
    elif isinstance(con, CEQC):
        return CIEQC(SR=con.SamplingRegion, partial=con.partial, factors=con.factors,
                     maxValue=ub, minValue=lb)
    else:
        raise ValueError("'con' must be an instance of 'DEQC' or 'CEQC'.")


def region_sampler(region: SamplingRegion,
                   func: Callable,
                   batch_size: int = 1000) -> Generator:
    """Returns a generator that yields 'batch_size' many samples of the region per call."""

    def data_loader():
        while True:
            s = region.drawTestSamples(batch_size)
            yield s, func(s)

    return data_loader()


def shuffle(inputs: Array['N,M', float],
            outputs: Array['N,K', float]) -> (Array['N,M', float], Array['N,K', float]):
    """Jointly shuffles the input and output array."""
    inputs_r = np.zeros((inputs.shape[0], inputs.shape[1]))
    outputs_r = np.zeros((outputs.shape[0], outputs.shape[1]))

    for i, m in enumerate(np.random.permutation(inputs.shape[0])):
        inputs_r[i, :] = inputs[m, :]
        outputs_r[i, :] = outputs[m, :]

    return inputs_r, outputs_r


def sigmoid_coeffs(degree: int) -> Array['N,1', int]:
    """Returns coefficients of the 'degree'th derivative of the sigmoid function."""
    if degree > 11:
        raise ValueError("derivatives of degree 11 or higher lead to overflowing coefficients")
    if degree < 0:
        raise ValueError("degree must be greater than or eqqal to zero")

    c = np.zeros(degree + 1, dtype=int)
    c[0] = 1
    for i in range(1, degree + 1):
        for j in range(i, -1, -1):
            c[j] = -j * c[j - 1] + (j + 1) * c[j]
    return np.append(c[::-1], 0)


def sigmoid(val: Array['N,M', float], degree: int = 0) -> Array['N,M', float]:
    """Evaluates the sigmoid function or its 'degree'th derivative at 'val'."""
    return np.polyval(sigmoid_coeffs(degree), 1. / (1 + np.exp(-val)))


def check_array(x):
    """Inefficient way of checking whether 'x' is an np.array."""
    try:
        x.shape
        return True
    except:
        return False


def stack(arrays: List, order='v'):
    """Wrapper for np.(v/h)stack that skips empty elements in 'arrays'."""
    if order not in ['v', 'h']:
        raise ValueError()

    filled_arrays = []
    for a in arrays:
        if a.size != 0:
            filled_arrays.append(a)

    if len(filled_arrays) == 0:
        return []

    if len(filled_arrays) == 1:
        return filled_arrays[0]

    if order == 'v':
        return np.vstack(filled_arrays)
    if order == 'h':
        return np.hstack(filled_arrays)


def make_grid_from_seq(seq: Array, sX: int, sY: int) -> Array:
    """Turns a (N*N, 2) array into two (N,N) grids."""
    U = np.zeros((sX, sY))
    count = 0
    for i in range(sX):
        for j in range(sY):
            U[i, j] = seq[count]
            count = count + 1
    return U


def make_seq_from_grid(X: Array, Y: Array) -> Array:
    """Turns to (N,N) grids into one (N*N, 2) array."""
    sX, sY = X.shape
    seq = []
    for i in range(sX):
        for j in range(sY):
            seq.append([X[i, j], Y[i, j]])
    return np.array(seq)


def combine_funcs(*funcs):
    def combined_func(*args, **kwargs):
        for f in funcs:
            f(*args, **kwargs)

    return combined_func


def smooth_trajectory(X,
                      Y,
                      ntraj,
                      ndim=1,
                      norder=1,
                      window=11,
                      dt=1e-2):

    # find breaking points
    if ntraj == 1:
        idc = [X.shape[0]]
    else:
        dist = np.linalg.norm(X[:, :ndim], axis=1)
        idc = np.argsort(np.diff(dist, axis=0))[-ntraj + 1:] + 1
        idc = np.append(idc, [X.shape[0]])
        idc = np.sort(idc)

    X_smooth = np.zeros(X.shape)
    Y_smooth = np.zeros(Y.shape)

    prev_idx = 0
    for idx in idc:
        t = np.linspace(0, 1, idx - prev_idx)

        for i in range(ndim):
            _, traj, _ = poly_regression_filter(t, X[prev_idx: idx, i], degree=1, window=window)
            X_smooth[prev_idx: idx, i] = traj

            for j in range(norder):
                deriv = np.diff(traj, axis=0) / dt
                deriv = np.append(deriv, [0])
                _, traj, _ = poly_regression_filter(t, deriv, degree=1, window=window)

                Y_smooth[prev_idx: idx, j * ndim + i] = traj
                if j < norder - 1:
                    X_smooth[prev_idx: idx, (j + 1) * ndim + i] = traj
        prev_idx = idx
    return X_smooth, Y_smooth


def integrate(model: BaseLearner, x0,
              t_max=10,
              lb=None,
              ub=None,
              method='single-step',
              dt=0.01,
              eps=1e-3,
              verbose=0,
              order=1):
    if method not in ['single-step', 'runge-kutta']:
        raise ValueError("Methods must be in ['single-step', 'runge-kutta']")

    x0 = np.reshape(x0, (1, -1))
    ndim = x0.shape[1]

    lb = np.ones(x0.shape) * -np.inf if lb is None else np.reshape(lb, x0.shape)
    ub = np.ones(x0.shape) * np.inf if ub is None else np.reshape(ub, x0.shape)

    p = []
    x = deepcopy(x0)
    t = 0
    while np.linalg.norm(x) > eps:
        t += dt
        p.append(deepcopy(x))
        grad = model.apply(x)

        if verbose == 1:
            print(f"{t:.2f}s: {x[0]} (x), {grad[0]} (dx)")

        for i in range(order - 1):
            x[0, i * ndim: (i + 1) * ndim] += dt * x[0, (i + 1) * ndim: (i + 2) * ndim]
        x[0, (order - 1) * ndim:] += dt * grad[0, :]

        if (x > ub).any() or (x < lb).any():
            break
        if t > t_max:
            break

    return np.reshape(np.array(p), (-1, ndim)), t


def poly_regression_filter(x,
                           y,
                           degree=1,
                           window=4,
                           closed=True):
    num_points = x.shape[0]
    left_offset = int(np.floor((window - 1) / 2))
    right_offset = int(np.ceil((window - 1) / 2))
    filtered = np.zeros(num_points)

    for i in range(num_points):
        left_idx = max([i - left_offset, 0])
        right_idx = min([i + right_offset + 1, num_points - 1])

        x_ = x[left_idx: right_idx]
        y_ = y[left_idx: right_idx]

        if i - left_offset < 0:
            for _ in range(left_offset - i):
                x_ = np.append([x[0]], x_)
                y_ = np.append([y[0]], y_)
        if i + right_offset + 2 > num_points:
            for _ in range(i + right_offset - num_points + 2):
                x_ = np.append(x_, [x[-1]])
                y_ = np.append(y_, [y[-1]])

        p = np.polyfit(x_, y_, degree)
        filtered[i] = np.polyval(p, x[i])

    if closed:
        return x, filtered, p
    else:
        return x[left_offset: num_points - right_offset], \
            filtered[left_offset: num_points - right_offset], \
            p


def swept_area_errror(X, fX):
    # resample
    num_points = fX.shape[0]
    num_dim = fX.shape[1]

    t_new = np.linspace(0, 1, num=num_points)

    t = np.linspace(0, 1, num=X.shape[0])
    spline = [interp1d(t, X[:, i], kind='cubic') for i in range(num_dim)]
    X_new = np.array([spline[i](t_new) for i in range(num_dim)]).T

    t = np.linspace(0, 1, num=fX.shape[0])
    spline = [interp1d(t, fX[:, i], kind='cubic') for i in range(num_dim)]
    fX_new = np.array([spline[i](t_new) for i in range(num_dim)]).T

    # calculate swept area error
    def area(u, v):
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        return norm_u * np.sin(np.arccos((u @ v) / (norm_v * norm_u)))

    return np.mean(
        [area(X_new[i + 1, :] - X_new[i, :], fX_new[i + 1, :] - fX_new[i, :]) for i in range(num_points - 1)])


def plot_history(history):
    loss_keys = list(history.keys())
    loss_keys.remove('constraint_metrics')

    plt.figure()
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    for key in loss_keys:
        plt.plot(history[key], label=key)
    plt.legend()

    constraints = history['constraint_metrics']
    plt.figure()
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Max. abs. violation')
    labels = [elem['label'] for elem in constraints[0]]
    for i, label in enumerate(labels):
        plt.plot([row[i]['max'] for row in constraints], label=label)
    plt.legend()

    plt.figure()
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Mean. abs. violation')
    labels = [elem['label'] for elem in constraints[0]]
    for i, label in enumerate(labels):
        plt.plot([row[i]['mean'] for row in constraints], label=label)
    plt.legend()
    return

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Reliability [%]')
    labels = [elem['label'] for elem in constraints[0]]
    for i, label in enumerate(labels):
        plt.plot([row[i]['reliability'] for row in constraints], label=label)
    plt.legend()

__all__ = ['BaseLearner', 'ELM', 'CELM', 'CMLP']
__version__ = '3.1'
__author__ = 'Jannick Stranghöner'

import warnings
from abc import ABC, abstractmethod
import copy
import types
from time import time
from keyboard import is_pressed
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import sklearn.utils
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from torch.autograd.functional import jacobian, hessian
from torch.optim.lr_scheduler import StepLR
from cvxopt import matrix, solvers

from constraints import *
import utils
warnings.filterwarnings("ignore")
torch._C._debug_only_display_vmap_fallback_warnings(True)

# cvxopt solver config
solvers.options['show_progress'] = False
solvers.options['max_iters'] = int(1e9)
solvers.options['abstol'] = 1e-7
solvers.options['reltol'] = 1e-6
solvers.options['feastol'] = 1e-7


class BaseLearner(ABC):

    def __init__(self, inp_dim, out_dim):

        self.inp_dim = inp_dim
        self.out_dim = out_dim

        # normalization parameters
        self._inp_offset = None
        self._inp_scale = None
        self._inp_range = None
        self._out_offset = None
        self._out_scale = None
        self._out_range = None
        super(BaseLearner, self).__init__()

    def normalize(self, x=None, y=None, overwrite=False):
        if x is not None:
            if self._inp_offset is None or overwrite:
                self._inp_range = np.array([np.amin(x, axis=0), np.amax(x, axis=0)])
                self._inp_offset = np.array([np.mean(x, axis=0)]).T
                self._inp_scale = np.array([np.std(x, axis=0)]).T
            x = (x.T - self._inp_offset) / self._inp_scale
            x = x.T

        if y is not None:
            if self._out_offset is None or overwrite:
                self._out_range = np.array([np.amin(y, axis=0), np.amax(y, axis=0)])
                self._out_offset = np.array([np.mean(y, axis=0)]).T
                self._out_scale = np.array([np.std(y, axis=0)]).T
            y = (y.T - self._out_offset) / self._out_scale
            y = y.T

        # I hate this
        if x is None and y is not None:
            return y
        if x is not None and y is None:
            return x
        if x is not None and y is not None:
            return x, y

    def denormalize(self, x=None, y=None):
        if x is not None and self._inp_offset is None:
            raise ValueError('No known input range for denormalization, normalize first')
        if y is not None and self._out_offset is None:
            raise ValueError('No known output range for denormalization, normalize first')

        if x is not None:
            x = x.T * self._inp_scale + self._inp_offset
            x = x.T

        if y is not None:
            y = y.T * self._out_scale + self._out_offset
            y = y.T

        if x is None and y is not None:
            return y
        if x is not None and y is None:
            return x
        if x is not None and y is not None:
            return x, y

    @abstractmethod
    def init(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def apply(self):
        pass


class ELM(BaseLearner):

    def __init__(self,
                 inp_dim,
                 out_dim,
                 use_bias=True,
                 hid_dim=100,
                 bip=False, # batch intrinsic plasticity
                 reg=1e-6,
                 mu=0.3,
                 batch_size=10000,
                 normalize=False):

        if not isinstance(inp_dim, int) or inp_dim < 1:
            raise ValueError("'inp_dim' must be of type int and greater than 0")
        if not isinstance(out_dim, int) or out_dim < 1:
            raise ValueError("'out_dim' must be of type int and greater than 0")
        if not isinstance(bip, bool):
            raise ValueError("'bip' must be of type bool")
        if not isinstance(reg, (float, int)) or reg <= 0:
            raise ValueError("'reg' must be of type float and greater than 0")
        if not isinstance(mu, (float, int)):
            raise ValueError("'mu' must be of type float")
        if not isinstance(normalize, bool):
            raise ValueError("'normalize' must be of type bool")

        BaseLearner.__init__(self, inp_dim, out_dim)

        self.hid_dim = hid_dim  # number of nodes in the hidden layer
        self.reg = reg  # regularization parameter for ridge regression
        self.bip = bip  # flag for using Batch Intrinsic Plasticity (bip)
        self.mu = mu  # desired mean activity parameter for bip
        self.batch_size = batch_size  # batchwise
        self._normalize = normalize
        self._use_bias = use_bias
        self._initialized = False

        # weights from input layer to hidden layer
        self.input_weights = 2 * np.random.uniform(size=(self.hid_dim, self.inp_dim)) - np.ones((self.hid_dim, self.inp_dim))

        # weights from hidden layer to output layer
        if self._use_bias:
            self.out_weights = 2 * np.random.uniform(size=(self.hid_dim + 1, self.out_dim)) - np.ones(
                (self.hid_dim + 1, self.out_dim))
        else:
            self.out_weights = 2 * np.random.uniform(size=(self.hid_dim, self.out_dim)) - np.ones((self.hid_dim, self.out_dim))

        # slope and bias parameters of activation functions
        self.a = np.ones(self.hid_dim)
        self.b = 2 * np.random.uniform(size=self.hid_dim) - np.ones(self.hid_dim)

    def init(self, x):
        """Initialize batch intrinsic plasticity. Must be called before .train(...) and .apply(...)!"""
        if x is not None and not isinstance(x, np.ndarray):
            raise ValueError("'x' must either be None or a numpy array")
        if self.bip:
            if x is None:
                raise ValueError("Can only apply Batch Intrinic Plasticity when 'x' was passed")
            self.__bip(x)

        self._initialized = True

    def train(self, x, y):
        """Calculate least-squares output weights for x and y."""

        if not isinstance(x, np.ndarray):
            raise ValueError("'x' must be a numpy array")
        if not isinstance(y, np.ndarray):
            raise ValueError("'y' must be a numpy array")

        if not self._initialized:
            raise ValueError("Model is not initialized")

        if x.ndim == 1:
            x = np.reshape(x, (x.shape[0], 1))
        if y.ndim == 1:
            y = np.reshape(y, (y.shape[0], 1))
        num_samples = np.size(x, 0)

        if self._normalize:
            x = self.normalize(x, overwrite=True)

        if num_samples < self.batch_size:
            hs = self.__calc_hidden_state(x)
            if self._use_bias: 
                hs = np.column_stack([hs, np.ones([hs.shape[0], 1])])

            if self._use_bias:
                g = np.linalg.inv(np.dot(hs.T, hs) + self.reg * np.identity(self.hid_dim + 1))
            else:
                g = np.linalg.inv(np.dot(hs.T, hs) + self.reg * np.identity(self.hid_dim))

            self.out_weights = np.dot(g, np.dot(hs.T, y))
        else:
            if self._use_bias:
                hth = np.zeros((self.hid_dim + 1, self.hid_dim + 1))
                hty = np.zeros((self.hid_dim + 1, self.out_dim))
            else:
                hth = np.zeros((self.hid_dim, self.hid_dim))
                hty = np.zeros((self.hid_dim, self.out_dim))

            num_batches = num_samples // self.batch_size
            rest = num_samples % self.batch_size
            for n in range(num_batches):
                h = self.__calc_hidden_state(x[n * self.batch_size:(n + 1) * self.batch_size, :])
                if self._use_bias: 
                    h = np.column_stack([h, np.ones([h.shape[0], 1])])
                hth += np.dot(h.T, h)
                hty += np.dot(h.T, y[n * self.batch_size:(n + 1) * self.batch_size, :])

            if rest > 0:
                h = self.__calc_hidden_state(x[num_batches * self.batch_size:-1, :])
                if self._use_bias: 
                    h = np.column_stack([h, np.ones([h.shape[0], 1])])
                hth += np.dot(h.T, h)
                hty += np.dot(h.T, y[num_batches * self.batch_size:-1, :])

            if self._use_bias:
                hth += self.reg * np.identity(self.hid_dim + 1)
            else:
                hth += self.reg * np.identity(self.hid_dim)

            self.out_weights = np.dot(np.linalg.pinv(hth), hty)

    def apply(self, x):
        """Runs inference for input array x, batch_size still applies."""

        if not isinstance(x, np.ndarray):
            raise ValueError("'x' must be a numpy array")

        if not self._initialized:
            raise ValueError("Model is not initialized")

        if x.ndim == 1:
            x = np.reshape(x, (x.shape[0], 1))

        if self._normalize:
            x = self.normalize(x, overwrite=False)

        num_samples = np.size(x, 0)
        if num_samples < self.batch_size:
            hs = self.__calc_hidden_state(x)
            if self._use_bias: hs = np.column_stack([hs, np.ones([hs.shape[0], 1])])
            y = np.dot(hs, self.out_weights)
        else:
            y = np.zeros((num_samples, self.out_dim))
            num_batches = num_samples // self.batch_size
            rest = num_samples % self.batch_size

            for n in range(num_batches):
                h = self.__calc_hidden_state(x[n * self.batch_size:(n + 1) * self.batch_size, :])
                if self._use_bias: 
                    h = np.column_stack([h, np.ones([h.shape[0], 1])])
                y[n * self.batch_size:(n + 1) * self.batch_size, :] = np.dot(h, self.out_weights)

            if rest > 0:
                h = self.__calc_hidden_state(x[num_batches * self.batch_size:-1, :])
                if self._use_bias: 
                    h = np.column_stack([h, np.ones([h.shape[0], 1])])
                y[num_batches * self.batch_size:-1, :] = np.dot(h, self.out_weights)

        if self._normalize:
            y = self.denormalize(y=y)
        self.out = y
        
        return self.out

    def __calc_hidden_state(self, x):
        g = x.dot(self.input_weights.T)
        return utils.sigmoid(self.a * g + self.b)

    def __bip(self, x):
        x = x if utils.check_array(x) else np.asarray(x)

        num_samples = np.size(x, 0)
        g = x.dot(self.input_weights.T)

        for hn in range(self.hid_dim):
            targets = np.random.exponential(self.mu, size=num_samples)
            hightars = np.size(targets[(targets >= 1) | (targets < 0)], 0)
            while hightars > 0:
                furthertargets = np.random.exponential(self.mu, size=hightars)
                targets = np.concatenate((targets[(targets < 1) & (targets >= 0)], furthertargets))
                hightars = np.size(targets[(targets >= 1) | (targets < 0)], 0)

            targets = np.sort(targets)
            s = np.sort(g[:, hn])

            Phi = np.column_stack([s, np.ones(num_samples)])
            targetsinv = np.array([-np.log(1. / targets - 1)])
            w = np.linalg.pinv(Phi).dot(targetsinv.T)

            self.a[hn] = w[0]
            self.b[hn] = w[1]


class CELM(ELM):
    """Implementation of an extreme learning machine with output constraints.

        Weighted sums of arbitrary partial derivatives of the output nodes can be
        constrained by adding subclasses of 'BaseConstraint' during construction.
        By freezing all layers except the last layer, the constrained optimization problem
        reduces to a linearly constrained quadratic optimization problem (quadratic programming, QP)
        w.r.t to the output weights, which can be solved using interior-point methods. To ensure
        continuous constraints hold, sampling and iteratively recomputing the QP-solution are applied
        until a predefined percentage of samples fulfill the constraint or a maximum number of iterations
        is reached.


        Parameters:
        - inp_dim (int): The dimensionality of the input layer.
        - out_dim (int): The dimensionality of the output layer.
        - hid_dim (int, optional): The number of neurons in the hidden layer. Defaults to 30.
        - max_iter (int, optional): The maximum number of SQPs for continuous constraint implementation . Defaults to 1000.
        - batch_size (int, optional): The size of the batch used in training. Defaults to 10000.
        - reg (float, optional): The weight of L2 regularization to prevent overfitting. Defaults to 1e-6.
        - mu (float, optional): The activity level for batch intrinsic plasticity (see bip). Defaults to 0.3.
        - eps (float, optional): The relaxation parameter for equality constraint checks. Defaults to 1e-3.
        - bip (bool, optional): Flag to use batch intrinsic plasticity, similar to batch normalization, for improved training stability. Defaults to False.
        - normalize (bool, optional): Flag to normalize input data, potentially improving model performance. Defaults to False.
        - verbose (int, optional): The frequency of console output during training, for monitoring progress in (0,1,2). Defaults to 2 (highest frequency).
        - cieqcs (np.array, optional): An array of Continuous Inequality Constraint (CIEQC) instances for the network. Defaults to an empty array.
        - ceqcs (np.array, optional): An array of Continuous Equality Constraint (CEQC) instances for the network. Defaults to an empty array.
        - dieqcs (np.array, optional): An array of Discrete Inequality Constraint (DIEQC) instances for the network. Defaults to an empty array.
        - deqcs (np.array, optional): An array of Discrete Equality Constraint (DEQC) instances for the network. Defaults to an empty array.
        - obj_fcts (np.array, optional): An array of objective functions, such as to solve Differential Equations. Defaults to an empty array.
    """

    def __init__(self, inp_dim: int, out_dim: int,
                 hid_dim: int = 30,
                 max_iter=1000,
                 batch_size: int = 10000,
                 reg: float = 1e-6,
                 mu: float = 0.3,
                 eps: float = 1e-3,
                 bip: bool = False,
                 normalize: bool = False,
                 verbose: int = 2,
                 cieqcs=np.array([]), 
                 ceqcs=np.array([]), 
                 dieqcs=np.array([]), 
                 deqcs=np.array([]), 
                 obj_fcts=np.array([])):
        ELM.__init__(self, inp_dim, out_dim, True, hid_dim, bip, reg, mu, batch_size, normalize)

        self.verbose = verbose
        self.hid_dim = hid_dim
        self._eps = eps

        self._cieqcs = list(cieqcs)
        self._ceqcs = list(ceqcs)
        self._dieqcs = list(dieqcs)
        self._deqcs = list(deqcs)
        self._obj_fcts = list(obj_fcts)
        self.max_iter = max_iter

        self._aeq = np.array([])
        self._beq = np.array([])
        self._aieq = np.array([])
        self._bieq = np.array([])

    def init(self, x=None):
        """Initializes constraint test values and dimensionalities. Must be called prior to .train(...) and .apply(...)!"""

        if x is not None and not isinstance(x, np.ndarray):
            raise ValueError("'x' must either be None or a numpy array")
        if self.bip and x is None:
            raise ValueError("Can only apply Batch Intrinic Plasticity when 'x' was passed")
        ELM.init(self, x)

        num_objfct = len(self._obj_fcts)
        for i in range(num_objfct):
            self._obj_fcts[i].inp_dim = self.inp_dim

        num_dieqs = len(self._dieqcs)
        for i in range(num_dieqs):
            self._dieqcs[i].inp_dim = self.inp_dim

        num_deqs = len(self._deqcs)
        for i in range(num_deqs):
            self._deqcs[i].inp_dim = self.inp_dim

        num_cieqs = len(self._cieqcs)
        for i in range(num_cieqs):
            self._cieqcs[i].inp_dim = self.inp_dim
            if self._cieqcs[i].max_test_value is None:
                self._cieqcs[i].max_test_value = self._cieqcs[i].max_value
            if self._cieqcs[i].min_test_value is None:
                self._cieqcs[i].min_test_value = self._cieqcs[i].min_value

        num_ceqs = len(self._ceqcs)
        for i in range(num_ceqs):
            self._ceqcs[i].inp_dim = self.inp_dim
            if self._ceqcs[i].max_test_value is None:
                self._ceqcs[i].max_test_value = self.__get_eq_test_value(self._ceqcs[i].value, self._eps)
            if self._ceqcs[i].min_test_value is None:
                self._ceqcs[i].min_test_value = self.__get_eq_test_value(self._ceqcs[i].value, -self._eps)

    def grad(self, u):
        return -np.array([-self.__fct(u, [[i]], [1]) for i in range(self.inp_dim)]).T

    def train(self, x, y):
        """Runs sequential quadratic program to minimize || self(x)-y || and respect possibly continuous constraints."""

        if not isinstance(x, np.ndarray):
            raise ValueError("'x' must be a numpy array")
        if not isinstance(y, np.ndarray):
            raise ValueError("'y' must be a numpy array")

        if not self._initialized:
            raise ValueError("Model is not initialized")

        mse = []
        reliab = []
        if x.ndim == 1:
            x = np.reshape(x, (x.shape[0], 1))
        if y.ndim == 1:
            y = np.reshape(y, (y.shape[0], 1))

        if self._normalize:
            x = self.normalize(x, overwrite=True)

        num_cieqcs = len(self._cieqcs)
        num_ceqcs = len(self._ceqcs)

        if self._normalize:
            for con in self._dieqcs:
                con.u = self.normalize(np.atleast_2d(con.u))
            for con in self._deqcs:
                con.u = self.normalize(np.atleast_2d(con.u))

        self.__clear_constraints()
        self.__fill_ieq_matrix(self._dieqcs)
        self.__fill_eq_matrix(self._deqcs)
        self.__qp(x, y)

        if num_cieqcs + num_ceqcs != 0:

            for i in range(self.max_iter):

                if is_pressed('esc'):
                    print("Constrained learning was interrupted manually")
                    break

                mse.append(self.__mse(x, y))

                # constraint fulfillment check
                test_samples_cieqcs = [[] for _ in range(num_cieqcs)]
                violations_cieqcs = [0 for _ in range(num_cieqcs)]
                ub_cieqcs = [0 for _ in range(num_cieqcs)]
                lb_cieqcs = [0 for _ in range(num_cieqcs)]

                test_samples_ceqcs = [[] for _ in range(num_ceqcs)]
                violations_ceqcs = [0 for _ in range(num_ceqcs)]
                ub_ceqcs = [0 for _ in range(num_ceqcs)]
                lb_ceqcs = [0 for _ in range(num_ceqcs)]

                # draw samples
                for c, con in enumerate(self._cieqcs):
                    test_samples_cieqcs[c] = con.draw_test_samples()
                    if self._normalize:
                        test_samples_cieqcs[c] = self.normalize(test_samples_cieqcs[c])
                for c, con in enumerate(self._ceqcs):
                    test_samples_ceqcs[c] = con.draw_test_samples()
                    if self._normalize:
                        test_samples_ceqcs[c] = self.normalize(test_samples_ceqcs[c])

                # check inequality constraints
                violationcount_cieqcs = np.zeros(num_cieqcs)
                violationcount_ceqcs = np.zeros(num_ceqcs)
                for c, con in enumerate(self._cieqcs):
                    violations_cieqcs[c] = 0
                    samples = self.__check_sample_dim(test_samples_cieqcs[c])
                    
                    # calculate constraint function value
                    for idx, pd in enumerate(con.partials):
                        violations_cieqcs[c] += self.__fct(samples, pd, con.factors[idx])
                        
                    # get bounds
                    ub_cieqcs[c] = self.__get_bound(con.max_test_value, samples)
                    lb_cieqcs[c] = self.__get_bound(con.min_test_value, samples)
                    
                    # count violations
                    violationcount_cieqcs[c] = np.sum((violations_cieqcs[c] > ub_cieqcs[c]) |
                                                      (violations_cieqcs[c] < lb_cieqcs[c]))

                # check equality constraints
                for c, con in enumerate(self._ceqcs):
                    violations_ceqcs[c] = 0
                    samples = self.__check_sample_dim(test_samples_ceqcs[c])

                    # calculate constraint function value
                    for idx, pd in enumerate(con.partials):
                        violations_ceqcs[c] = self.__fct(samples, pd, con.factors[idx])

                    # get bounds
                    ub_ceqcs[c] = self.__get_bound(con.max_test_value, samples)
                    lb_ceqcs[c] = self.__get_bound(con.min_test_value, samples)

                    # count violations
                    violationcount_ceqcs[c] = np.sum((violations_ceqcs[c] > ub_ceqcs[c]) |
                                                     (violations_ceqcs[c] < lb_ceqcs[c]))

                # cmd output
                cmd_output = "Iter. (" + str(i + 1) + "): "
                _reliability = []
                if num_cieqcs > 0:
                    cmd_output += "IEQs: "
                    for c in range(num_cieqcs - 1):
                        nInputs = len(self._cieqcs[c].u)
                        reliability = 1 - (violationcount_cieqcs[c] / self._cieqcs[c].test_samples_per_iteration)
                        cmd_output += f"Rel. ({nInputs}): {reliability:.3f}, "
                        _reliability.append( ('CIEQC_'+str(c), reliability))
                    nInputs = len(self._cieqcs[num_cieqcs - 1].u)
                    reliability = 1 - (violationcount_cieqcs[num_cieqcs - 1] / self._cieqcs[
                        num_cieqcs - 1].test_samples_per_iteration)
                    cmd_output += f"Rel. ({nInputs}): {reliability:.3f}"
                    _reliability.append(('CIEQC_' + str(num_cieqcs - 1), reliability))

                if num_ceqcs > 0:
                    cmd_output += "EQs: "
                    for c in range(num_ceqcs - 1):
                        nInputs = len(self._ceqcs[c].u)
                        reliability = 1 - (violationcount_ceqcs[c] / self._ceqcs[c].test_samples_per_iteration)
                        cmd_output += f"Rel. ({nInputs}): {reliability:.3f}, "
                        _reliability.append(('CEQC_' + str(c), reliability))
                    nInputs = len(self._ceqcs[num_ceqcs - 1].u)
                    reliability = 1 - (violationcount_ceqcs[num_ceqcs - 1] / self._ceqcs[
                        num_ceqcs - 1].test_samples_per_iteration)
                    cmd_output += f"Rel. ({nInputs}): {reliability:.3f}"
                    _reliability.append(('CEQC_' + str(num_cieqcs - 1), reliability))
                reliab.append(_reliability)

                # break test
                test_samples_per_iteration_cieqcs = [con.test_samples_per_iteration for con in self._cieqcs]
                test_samples_per_iteration_ceqcs = [con.test_samples_per_iteration for con in self._ceqcs]
                satisfaction_thresholds_cieqcs = [con.satisfaction_threshold for con in self._cieqcs]
                satisfaction_thresholds_ceqcs = [con.satisfaction_threshold for con in self._ceqcs]

                flag_StopSampling_cieqcs = (1 - np.divide(violationcount_cieqcs,
                                                          test_samples_per_iteration_cieqcs)) > satisfaction_thresholds_cieqcs
                flag_StopSampling_ceqcs = (1 - np.divide(violationcount_ceqcs,
                                                         test_samples_per_iteration_ceqcs)) > satisfaction_thresholds_ceqcs
                if sum(flag_StopSampling_cieqcs) == num_cieqcs and sum(flag_StopSampling_ceqcs) == num_ceqcs:
                    break

                # add most violating points (inequalities)
                for c, con in enumerate(self._cieqcs):
                    if flag_StopSampling_cieqcs[c] == 0:

                        viol = violations_cieqcs[c]
                        ub = ub_cieqcs[c]
                        lb = lb_cieqcs[c]

                        if all(ub == np.inf):
                            viol = -(viol - lb)
                        elif all(lb == -np.inf):
                            viol -= ub
                        else:
                            viol = np.maximum(viol - ub, -(viol - lb))
                        sortval = np.sort(viol)[::-1]
                        sortind = np.argsort(viol)[::-1]

                        j = 1
                        counter = 0
                        test_samples_temp = test_samples_cieqcs[c][sortind, :]
                        while j < len(test_samples_temp) and counter < con.samples_per_iteration:
                            if sortval[j] > 0:
                                con.u = utils.stack([con.u, test_samples_temp[j, :]], 'v')
                                counter += 1
                            j += 1

                # add most violating points (equalities)
                for c, con in enumerate(self._ceqcs):
                    if flag_StopSampling_ceqcs[c] == 0:

                        viol = violations_ceqcs[c]
                        ub = ub_ceqcs[c]
                        lb = lb_ceqcs[c]

                        if all(ub == np.inf):
                            viol = -(viol - lb)
                        elif all(lb == -np.inf):
                            viol -= ub
                        else:
                            viol = np.maximum(viol - ub, -(viol - lb))
                        sortval = np.sort(viol)[::-1]
                        sortind = np.argsort(viol)[::-1]

                        j = 1
                        counter = 0
                        test_samples_temp = test_samples_ceqcs[c][sortind, :]
                        while j < len(test_samples_temp) and counter < con.samples_per_iteration:
                            if sortval[j] > 0:
                                con.u = utils.stack([con.u, test_samples_temp[j, :]], 'v')
                                counter += 1
                            j += 1

                # constraint learning
                self.__clear_constraints()
                self.__fill_ieq_matrix(self._cieqcs)
                self.__fill_eq_matrix(self._ceqcs)
                self.__fill_ieq_matrix(self._dieqcs)
                self.__fill_eq_matrix(self._deqcs)

                try:
                    cmd_output += self.__qp(x, y)
                except ValueError as e:
                    print(e)
                    break

                if self.verbose == 1:
                    print("Iteration (" + str(i + 1) + ")")
                elif self.verbose == 2:
                    print(cmd_output)
        else:
            i = 0
        print("Learning finished")

        return {
            'iter': i,
            'mse': mse,
            'reliability': reliab
        }

    def __qp(self, x, y):
        """Solve quadratic program subject to constraints defined by self.aieq, self.aeq, self.bieq and self.beq."""

        if not isinstance(x, np.ndarray):
            raise ValueError("'x' must be a numpy array")
        if not isinstance(y, np.ndarray):
            raise ValueError("'y' must be a numpy array")

        d = y.flatten('F')
        h, _ = self.__objective_function(x, y)
        xH = h.shape[0]
        yH = h.shape[1]

        for i in range(self.out_dim):
            if i == 0:
                C = np.hstack([np.zeros((xH, i * yH)), h, np.zeros((xH, (self.out_dim - i - 1) * yH))])
            else:
                C = np.vstack([C, np.hstack([np.zeros((xH, i * yH)), h, np.zeros((xH, (self.out_dim - i - 1) * yH))])])

        C = np.vstack([C, np.sqrt(self.reg) * np.eye((self.hid_dim + 1) * self.out_dim)])
        d = np.hstack([np.array([d]), np.zeros((1, (self.hid_dim + 1) * self.out_dim))]).T

        P = matrix(C.T.dot(C))
        q = matrix(-1 * C.T.dot(d))

        no_eqs = (self._aeq.size == 0)
        no_ieqs = (self._aieq.size == 0)

        # trf constraint matrices
        if not no_eqs:
            inv = 1 / self._beq
            mask = (inv == np.inf) | (inv == -np.inf)
            Aeq = copy.deepcopy(self._aeq)
            beq = copy.deepcopy(self._beq)
            is_negative = beq < 0
            Aeq[~mask] /= np.reshape(beq[~mask], (-1, 1))
            beq[~mask] = 1.
            Aeq[is_negative] *= -1.
            beq[is_negative] *= -1.
        if not no_ieqs:
            inv = 1 / self._bieq
            mask = (inv == np.inf) | (inv == -np.inf)
            Aieq = copy.deepcopy(self._aieq)
            bieq = copy.deepcopy(self._bieq)
            is_negative = bieq < 0
            Aieq[~mask] /= np.reshape(bieq[~mask], (-1, 1))
            bieq[~mask] = 1.
            Aieq[is_negative] *= -1.
            bieq[is_negative] *= -1.

        # parameterize and run solver
        if no_eqs and no_ieqs:
            sol = solvers.qp(P, q)
        elif no_eqs:
            Aieq = matrix(Aieq)
            bieq = matrix(bieq)
            sol = solvers.qp(P, q, G=Aieq, h=bieq)
            self.out_weights = np.reshape(sol['x'], self.out_weights.shape, order='F')
        elif no_ieqs:
            Aeq = matrix(Aeq)
            beq = matrix(beq)
            sol = solvers.qp(P, q, A=Aeq, b=beq)
        else:
            Aieq = matrix(Aieq)
            bieq = matrix(bieq)
            Aeq = matrix(Aeq)
            beq = matrix(beq)
            sol = solvers.qp(P, q, G=Aieq, h=bieq, A=Aeq, b=beq)#, solver='mosek')

        self.out_weights = np.reshape(sol['x'], self.out_weights.shape, order='F')
        return f" | Solution [{sol['status']}],  MSE: {self.__mse(x, y)}, MAE: {self.__mae(x, y)}"

    def __objective_function(self, x, y):
        """Fill hidden state matrix according to list of linear objective functions"""

        if not isinstance(x, np.ndarray):
            raise ValueError("'x' must be a numpy array")
        if not isinstance(y, np.ndarray):
            raise ValueError("'y' must be a numpy array")

        if not self._obj_fcts:
            h = self._ELM__calc_hidden_state(x)
            return (utils.stack((h, np.ones((h.shape[0], 1))), 'h'), y)

        input_weights = self.input_weights.T
        a = self.a.T
        b = self.b.T

        Heff = np.array([])
        Yeff = np.array([])
        for of, objfct in enumerate(self._obj_fcts):
            num_samples = x.shape[0]
            if num_samples == 0:
                return

            partials = objfct.partials
            coeffs = objfct.factors
            
            for idx, pd in enumerate(partials):
                HTemp = np.array([])
                factors = coeffs[idx]
                if callable(factors):
                    factors = factors(x)
                else:
                    factors = np.repeat(factors, num_samples)

                for s in range(self.out_dim):
                    D = len(pd[s])
                    matTemp = np.ones((self.hid_dim, num_samples))
                    for p in pd[s]:
                        partial_input_weights = np.multiply(a, input_weights[p, :])
                        matTemp *= np.array([partial_input_weights]).T

                    hid_derivative = utils.sigmoid(a * (x.dot(input_weights)) + b, D) * matTemp.T
                    hid_derivative = utils.stack((hid_derivative, np.ones((hid_derivative.shape[0], 1))), 'h') if len(
                        pd[s]) == 0 else utils.stack((hid_derivative, np.zeros((hid_derivative.shape[0], 1))), 'h')
                    HTemp = utils.stack((HTemp, np.tile(factors[s], (self.hid_dim + 1, 1)).T * hid_derivative), 'h')

                if idx == 0:
                    h = HTemp
                else:
                    h += HTemp

            Heff = utils.stack((Heff, h))
            Yeff = utils.stack((Yeff, y[:, of]), 'h')
        return (Heff, Yeff)

    def __clear_constraints(self):
        self._aeq = np.array([])
        self._beq = np.array([])
        self._aieq = np.array([])
        self._bieq = np.array([])

    def __fill_ieq_matrix(self, cons):
        """Fill self.aieq and self.bieq matrices to parametrize the solver according to linear inequality constraints."""

        if len(cons) == 0:
            return

        input_weights = self.input_weights.T
        a = self.a.T
        b = self.b.T

        for con in cons:
            partials = con.partials
            coeffs = con.factors

            if con.min_value == -np.inf and con.max_value == np.Inf:
                continue

            u = self.__check_sample_dim(np.asarray(con.u))
            if u.size == 0:
                continue
            ub = self.__get_bound(con.max_test_value, u)
            lb = self.__get_bound(con.min_test_value, u)

            num_samples = u.shape[0]

            if all(ub == np.inf) and all(lb == -np.inf):
                continue

            for idx, pd in enumerate(partials):
                AieqMinTemp = np.array([])
                bieqMinTemp = np.zeros((num_samples,))
                AieqMaxTemp = np.array([])
                bieqMaxTemp = np.zeros((num_samples,))

                factors = coeffs[idx]
                if callable(factors):
                    factors = factors(u)
                else:
                    factors = factors

                for s in range(self.out_dim):
                    D = len(pd[s])
                    matTemp = np.ones((self.hid_dim, num_samples))
                    for p in pd[s]:
                        partial_input_weights = np.multiply(a, input_weights[p, :])
                        matTemp *= np.array([partial_input_weights]).T
                        if self._normalize:
                            matTemp /= self._inp_scale[p]

                    hid_derivative = utils.sigmoid(a * (u.dot(input_weights)) + b, D) * matTemp.T
                    hid_derivative = utils.stack((hid_derivative, np.ones((hid_derivative.shape[0], 1))), 'h') if len(
                        pd[s]) == 0 else utils.stack((hid_derivative, np.zeros((hid_derivative.shape[0], 1))), 'h')

                    if con.min_value != -np.inf:
                        AieqMinTemp = utils.stack(
                            (AieqMinTemp, np.tile(-factors[s], (self.hid_dim + 1, 1)).T * hid_derivative), 'h')
                        bieqMinTemp -= lb

                    if con.max_value != np.Inf:
                        AieqMaxTemp = utils.stack((AieqMaxTemp, np.tile(factors[s], (self.hid_dim + 1, 1)).T * hid_derivative),
                                            'h')
                        bieqMaxTemp += ub

                bieqMinTemp /= self.out_dim
                bieqMaxTemp /= self.out_dim

                if idx == 0:
                    AieqTempSum = utils.stack((AieqMinTemp, AieqMaxTemp))
                    if con.max_value != np.Inf and con.min_value == -np.inf:
                        bieqTempSum = bieqMaxTemp
                    elif con.max_value == np.Inf and con.min_value != -np.inf:
                        bieqTempSum = bieqMinTemp
                    else:
                        bieqTempSum = utils.stack((bieqMinTemp, bieqMaxTemp), 'h')
                else:
                    AieqTempSum += utils.stack((AieqMinTemp, AieqMaxTemp))
                    if con.max_value != np.Inf and con.min_value == -np.inf:
                        bieqTempSum += bieqMaxTemp
                    elif con.max_value == np.Inf and con.min_value != -np.inf:
                        bieqTempSum += bieqMinTemp
                    else:
                        bieqTempSum += utils.stack((bieqMinTemp, bieqMaxTemp), 'h')

            self._aieq = np.reshape(utils.stack((self._aieq, AieqTempSum)), (-1, (self.hid_dim + 1) * self.out_dim))
            self._bieq = utils.stack([self._bieq, np.array([bieqTempSum]).flatten()], 'h')
            self._bieq = self._bieq.flatten()

    def __fill_eq_matrix(self, cons):
        """Fill self.aeq and self.beq matrices to parametrize the solver according to linear equality constraints."""

        if len(cons) == 0:
            return

        input_weights = self.input_weights.T
        a = self.a.T
        b = self.b.T

        for c, con in enumerate(cons):

            u = self.__check_sample_dim(np.asarray(con.u))
            if u.size == 0:
                continue
            tar = self.__get_bound(con.value, u)

            num_samples = u.shape[0]

            partials = con.partials
            coeffs = con.factors

            if u.size == 0:
                continue

            for idx, pd in enumerate(partials):
                AeqTemp = np.array([])
                beqTemp = 0

                factors = coeffs[idx]
                if callable(factors):
                    factors = factors(u)
                else:
                    factors = factors

                for s in range(self.out_dim):
                    D = len(pd[s])
                    matTemp = np.ones((self.hid_dim, num_samples))
                    for p in pd[s]:
                        partial_input_weights = np.multiply(a, input_weights[p, :])
                        matTemp *= np.array([partial_input_weights]).T
                        if self._normalize:
                            matTemp /= self._inp_scale[p]

                    hid_derivative = utils.sigmoid(a * (u.dot(input_weights)) + b, D) * matTemp.T
                    hid_derivative = utils.stack((hid_derivative, np.ones((hid_derivative.shape[0], 1))), 'h') if len(
                        pd[s]) == 0 else utils.stack((hid_derivative, np.zeros((hid_derivative.shape[0], 1))), 'h')

                    AeqTemp = utils.stack((AeqTemp, np.tile(-factors[s], (self.hid_dim + 1, 1)).T * hid_derivative), 'h')
                    beqTemp -= tar

                if idx == 0:
                    AeqTempSum = AeqTemp
                    beqTempSum = beqTemp
                else:
                    AeqTempSum = AeqTempSum + AeqTemp
                    beqTempSum = beqTempSum + beqTemp

            self._aeq = utils.stack((self._aeq, AeqTempSum))
            self._beq = np.append(self._beq, beqTempSum / self.out_dim)

    def __fct(self, x, partials, factors):
        """Calculate linear combination of partial derivatives of the hidden state."""

        input_weights = self.input_weights.T
        a = self.a.T
        b = self.b.T

        x = np.asarray(x) if not utils.check_array(x) else x

        if x.ndim == 1:
            num_samples = 1
            x = np.array([x])
        else:
            num_samples = x.shape[0]

        out_dim = len(partials)
        val = np.zeros((num_samples, out_dim))

        if callable(factors):
            facts = factors(x)
        else:
            facts = factors

        for s in range(out_dim):

            D = len(partials[s])
            matTemp = np.ones((self.hid_dim, num_samples))
            for p in partials[s]:
                partial_input_weights = np.multiply(a, input_weights[p, :])
                matTemp *= np.array([partial_input_weights]).T
                if self._normalize:
                    matTemp = np.multiply(matTemp, 1/self._inp_scale[p])

            hid_derivative = utils.sigmoid(a * (x.dot(input_weights)) + b.T, D) * matTemp.T
            hid_derivative = utils.stack((hid_derivative, np.ones((hid_derivative.shape[0], 1))), 'h') if len(
                partials[s]) == 0 else utils.stack((hid_derivative, np.zeros((hid_derivative.shape[0], 1))), 'h')

            valTemp = hid_derivative.dot(self.out_weights)

            val[:, s] = valTemp[:, s] * facts[s]

        return np.sum(val, axis=1)

    def __check_sample_dim(self, u):
        """Returns 2d numpy array"""
        if u.ndim == 1:
            return np.reshape(u, (1, u.shape[0]))
        else:
            return u

    def __get_eq_test_value(self, value, eps):
        """Relaxes test value for equality constraints."""
        if callable(value):
            return lambda x: value(x) + eps
        else:
            return value + eps

    def __get_bound(self, bound, u):
        """Returns bounds for each element in u (possibly as a function of u)"""
        if callable(bound):
            if self._normalize:
                u = self.denormalize(u)
            bound = np.reshape(bound(u), (-1,))
        else:
            bound = np.repeat(bound, u.shape[0])
        return bound

    def __mse(self, x, y):
        """L2 error with objective functions."""
        [Heff, Yeff] = self.__objective_function(x, y)
        if not self._obj_fcts:
            diff = Heff.dot(self.out_weights) - Yeff
            return np.mean(diff ** 2, axis=0)
        else:
            diff = Heff.dot(self.out_weights).T - Yeff
            return np.mean(diff ** 2, axis=1)

    def __mae(self, x, y):
        """L1 error with objective functions."""
        [Heff, Yeff] = self.__objective_function(x, y)
        if not self._obj_fcts:
            diff = Heff.dot(self.out_weights) - Yeff
            return np.mean(np.abs(diff), axis=0)
        else:
            diff = Heff.dot(self.out_weights).T - Yeff
            return np.mean(np.abs(diff), axis=1)


class CMLP(BaseLearner, nn.Module):
    CMD_LEVEL_BASIC = 2
    CMD_LEVEL_IMPORTANT = 1

    def __init__(self, inp_dim, out_dim,
                 loss=nn.MSELoss(),
                 verbose=2,
                 eps=1e-3,
                 tik=1e-4,
                 reg=1e-6,
                 normalize=False):

        # check input
        if not isinstance(verbose, int) or verbose not in (0, 1, 2):
            raise ValueError("'verbose' must be of type int and between 0 and 2")
        if eps < 0:
            raise ValueError("'eps' must be of type float and greater than 0")
        if tik < 0:
            raise ValueError("'tik' must be of type float and greater than 0")
        if reg is not None and reg < 0:  # can be none
            raise ValueError("'reg' must be of type float and greater than 0")
        if not isinstance(normalize, bool):
            raise ValueError("'normalize' must be of type bool")
        super().__init__(inp_dim, out_dim)

        # basic info and topology
        self.layers = nn.Sequential()
        self._total_constraint_weight = 1
        self._data_generator = None
        self._initialized = False
        self._epoch_metrics_buffer = dict()
        self._epoch_cout_buffer = ''
        self._epoch_sample_buffer = []
        self._epoch_sample_idc = []
        self._previous_epoch_sample_buffer = []

        # hyper-parameters
        self._normalize = normalize
        self._criterion = loss
        self._reg = reg
        self._tik = tik
        self._eps = eps
        self._verbose = verbose
        self._lambda = 0

        # matrices that define the constraints during the QP
        self._constraints = []
        self._discrete_constraints = []
        self._continuous_constraints = []
        self._ieq_constraints = []
        self._eq_constraints = []
        self._linear_constraints = []
        self._nonlinear_constraints = []
        self._aeq = np.array([])
        self._beq = np.array([])
        self._aieq = np.array([])
        self._bieq = np.array([])

    @property
    def device(self):
        """returns torch.device"""
        return next(self.parameters()).device

    def add_constraint(self, *args: BaseConstraint):
        """adds scalar or list of BaseConstraint subclass instance(s)"""
        for con in args:
            if not isinstance(con, BaseConstraint):
                raise ValueError('Each constraint must subclass BaseConstraint')
            con.inp_dim = self.inp_dim
            self._constraints.append(con)

            if isinstance(con, LinearConstraint):
                self._linear_constraints.append(con)
            if isinstance(con, NonlinearConstraint):
                self._nonlinear_constraints.append(con)
            if isinstance(con, EqualityConstraint):
                self._eq_constraints.append(con)
                max_test_value = self.__get_eq_test_value(con.value, con.eps) if con.max_test_value is None else con.max_test_value
                min_test_value = self.__get_eq_test_value(con.value, -con.eps) if con.min_test_value is None else con.min_test_value
            if isinstance(con, InequalityConstraint):
                self._ieq_constraints.append(con)
                max_test_value = self.__get_eq_test_value(con.max_value) if con.max_test_value is None else con.max_test_value
                min_test_value = self.__get_eq_test_value(con.min_value) if con.min_test_value is None else con.min_test_value
            if isinstance(con, DiscreteConstraint):
                self._discrete_constraints.append(con)
            if isinstance(con, ContinuousConstraint):
                self._continuous_constraints.append(con)
                con.max_test_value = max_test_value
                con.min_test_value = min_test_value

    def init(self, x=None, y=None, generator=None):
        """Initializes train data generator and constraints.
        Must be called before using "train(...)" or "apply(...)"!
        """

        if (x is None or y is None) and generator is None:
            raise ValueError('A data source must be specified either via a generator or an input array x and a target array y')
        if x is not None and generator is not None:
            raise ValueError('Only one data source should be specified')

        # turn input data into generator
        if x is not None:
            if x.ndim == 1:
                x = np.atleast_2d(x).T
            if y.ndim == 1:
                y = np.atleast_2d(y).T
            def g():
                while True:
                    yield x, y
            generator = g()

        if not isinstance(generator, types.GeneratorType):
            raise ValueError('not a generator')

        # label constraints for cmd output
        cnt = {cls: 1 for cls in [con.__class__.__name__ for con in self._constraints]}
        for con in self._constraints:
            if con.label == '':
                cls = con.__class__.__name__
                con.label = cls + '_' + str(cnt[cls])
                cnt[cls] = cnt[cls] + 1

        self._total_constraint_weight = sum([con.weight for con in self._constraints])
        self._data_generator = generator
        self._initialized = True

    def add_layer(self, layer):
        self.layers = nn.Sequential(*list(self.layers.children()), layer)

    def forward(self, x):
        return self.layers(x)

    def apply(self, x, to_numpy=True):
        """Does the forward pass of the model.

        Args:
            x (np.ndarray or torch.tensor): input of the forward pass. First dimension is always the number of samples
            to_numpy (bool, optional):

        """

        if not self._initialized:
            raise ValueError("Model is not initialized")

        if self._normalize:
            x = self.normalize(x, overwrite=False)

        # turn input into valid tensor
        if not torch.is_tensor(x):
            if not isinstance(x, np.ndarray):
                raise ValueError("'x' must be a numpy array or tensor")
            x = torch.from_numpy(x).float()
        x.requires_grad = True

        # resize array if needed
        if x.ndimension() != 2:
            x = x.view(x.size(), 1)
        if len(list(x.size())) == 1:
            x = x.view(x.shape[0], 1)

        # apply forward pass
        out = self.forward(x)
        if to_numpy:
            return out.detach().numpy()
        else:
            return out

    def jacobian(self, u):
        """Returns differentiable jacobian tensor of dimensionality (n_samples x n_out x n_in)."""
        if self._normalize:
            u = self.normalize(u)

        t_u = torch.from_numpy(u).float().to(self.device)
        t_u.requires_grad = True

        return self.__jac(t_u)

    def hessian(self, u):
        """Returns differentiable hessian tensor of dimensionality (n_samples x n_out x n_in x n_in)."""
        if self._normalize:
            u = self.normalize(u)
            t_inp_scale = torch.FloatTensor(self._inp_scale).to(self.device)[:, 0]

        t_u = torch.from_numpy(u).float().to(self.device)
        t_u.requires_grad = True

        hess = jacobian(lambda x: self.__jac(x).sum(0), t_u, vectorize=True, create_graph=True)\
            .transpose(0, 2)\
            .transpose(1, 2)
        if self._normalize:
            hess = hess / t_inp_scale[None, None, None, :]
        return hess

    def train(self, x, y, **kwargs):
        self.init(x, y)
        return self.train_with_composite_loss(**kwargs)

    def train_without_constraints(self, max_epoch=1000, optimizer="Adam", optim_params={},
                                  threshold=-np.inf, lr=None, batch_size=None, validation_ratio=None):

        params = self.__validate_train_params(constrained=False, max_epoch=max_epoch, lr=lr, threshold=threshold, batch_size=batch_size, validation_ratio=validation_ratio, optimizer=optimizer, optim_params=optim_params)
        return self.__train(params)

    def train_with_composite_loss(self, **kwargs):
        params = self.__validate_train_params(constrained=bool(self._constraints), **kwargs)
        return self.__train(params)

    def sqp(self, max_iter=1000,  # number of qp's
            skip_initial_qp=False,  #
            analytical=False,  # constraints are enforced through penalty term instead of hard constraints
            constraint_weight=1,  # weight of constraints if analytical is True
            constraint_weight_factor=1,  # by how much constraint_weight is increased each iterationy
            ieq_test_epsilon=None):
        """..."""
        # todo: better doc string

        if self._nonlinear_constraints:
            warnings.warn(f"'constrained_iterative_last_layer_qp' only works for linear constraints, " +
                          f"{len(self._nonlinear_constraints)} non-linear constraints are ignored.")

        if ieq_test_epsilon is None:
            ieq_test_epsilon = self._eps

        # await esc release if previous method was exited using esc
        while is_pressed('esc'):
            pass

        # tighten up inequality constraint to account for residuals
        if not analytical:
            backup_bounds = [(c.max_value, c.min_value) for c in self._ieq_constraints]
            for c in self._ieq_constraints:
                c.max_value = self.__get_eq_test_value(c.max_value, -ieq_test_epsilon)
                c.min_value = self.__get_eq_test_value(c.min_value, ieq_test_epsilon)

        history = []
        for i in range(max_iter):
            self._epoch_cout_buffer = ''
            self._epoch_metrics_buffer = dict()

            # calculate constraint fulfillment
            # this is skipped in the first iteration, unless skip_initial_qp is true
            if i != 0 or skip_initial_qp:
                break_flags = []

                self._epoch_metrics_buffer['constraint_metrics'] = [None for _ in cont_constraints]
                for c, con in enumerate(self._continuous_constraints):
                    rel, mean_viol, max_viol = self.__verify_and_resample_constraint(con)
                    break_flags.append(rel > con.satisfaction_threshold)

                    self._epoch_metrics_buffer['constraint_metrics'][c] = {
                        'mean': mean_viol,
                        'max': max_viol,
                        'reliability': rel,
                        'label': con.label
                    }
                    self._epoch_cout_buffer += f'     | {con.label}({len(con.u)}): ' \
                                               f'{rel*100:.2f}% [rel], ' \
                                               f'{mean_viol:.3f} [mean], ' \
                                               f'{max_viol:.3f} [max]\n'

            try:
                (x, y) = next(self._data_generator)
            except StopIteration:
                self.__stdout('- generator is empty -')
                break

            if self._normalize:
                x = self.normalize(x, overwrite=i == 0)

            if is_pressed('esc'):
                self.__stdout("Learning was interrupted manually")
                break

            # fill constraint matrices
            self.__clear_constraints()
            self.__fill_ieq_matrix()
            self.__fill_eq_matrix()

            # solve qp
            self._lambda = constraint_weight*constraint_weight_factor**i
            self._epoch_metrics_buffer['lambda'] = self._lambda
            solver_msg, terminated = self.__qp(x, y, analytical=analytical)

            if not terminated:
                history.append(copy.deepcopy(self._epoch_metrics_buffer))
                self.__stdout(solver_msg)
                break

            if True or no_continuous_constraints:
                history.append(copy.deepcopy(self._epoch_metrics_buffer))
                self.__stdout("Solved QP " + solver_msg)
                break

            self.__stdout("Iteration (" + str(i) + ")", exact_verbosity=CMLP.CMD_LEVEL_IMPORTANT)
            self.__stdout(f'Iter. ({i+int(skip_initial_qp)}): ' + solver_msg + self._epoch_cout_buffer, verbosity=CMLP.CMD_LEVEL_BASIC)

            if i != 0 or skip_initial_qp:
                history.append(copy.deepcopy(self._epoch_metrics_buffer))
                if all(break_flags):
                    break

        # restore tightened inequality bounds
        if not analytical:
            for c, (ub, lb) in zip(self._ieq_constraints, backup_bounds):
                c.max_value = ub
                c.min_value = lb

        return self.__reformat_history(history)

    def __train(self, params):
        while is_pressed('esc'):
            pass

        self._epoch_sample_buffer = [{'samples': [], 'violations': []} for _ in self._continuous_constraints]
        self._epoch_sample_idc = [[] for _ in self._continuous_constraints]
        optimizer = self.__get_optimizer(params)
        do_validation = params['validation_ratio'] is not None
        history = []
        epochs_since_last_verification = 0
        for epoch in range(params['max_epoch']):
            start = time()
            try:
                (x, y) = next(self._data_generator)
            except StopIteration:
                self.__stdout('- generator is empty -')
                break

            if self._normalize:
                x = self.normalize(x, overwrite=epoch == 0)

            t_x = torch.from_numpy(x).float().to(self.device)
            t_y = torch.from_numpy(y).float().to(self.device)
            t_x.requires_grad = True

            if do_validation:
                t_x, t_x_val, t_y, t_y_val = train_test_split(t_x, t_y,
                                                              test_size=params['validation_ratio'],
                                                              random_state=42)
            t_x, t_y = sklearn.utils.utils.shuffle(t_x, t_y)

            batch_size = x.shape[0] if params['batch_size'] is None else params['batch_size']  # None batch size
            batch_size = x.shape[0] if batch_size > x.shape[0] else batch_size  # batch size larger than dataset
            num_batches = x.shape[0] // batch_size

            self._lambda = params['lambda_max'] * utils.sigmoid((epoch - params['max_epoch'] / 2) / params['max_epoch'] * 12 + 1)
            if params['constrained'] and (epoch == 0 or epochs_since_last_verification >= params['resample_freq']):
                optimizer = self.__get_optimizer(params)  # warm restart
                epochs_since_last_verification = 0

                # evaluate constraint verification and draw fresh samples
                self.__resample_constraints(params)
                self._previous_epoch_sample_buffer = copy.deepcopy(self._epoch_sample_buffer)  # save metrics for later comp

                # increase resample frequency to allow convergence
                if epoch > 0:
                    params['resample_freq'] = params['resample_freq'] * 1.1
                    self._epoch_cout_buffer += f'New resample frequency: {int(params["resample_freq"])}'
                self.__stdout(verbosity=CMLP.CMD_LEVEL_IMPORTANT)
            else:
                epochs_since_last_verification += 1

            for batch in range(num_batches):

                # draw samples for each constraint depending on sampling strategy
                if params['constrained'] and params['sampling_strategy'] in ['uniform', 'adaptive']:
                    for i, con in enumerate(self._continuous_constraints):
                        samples = self._epoch_sample_buffer[i]['buffer']['samples']
                        if params['sampling_strategy'] == 'uniform':
                            p = np.ones([samples.shape[0], ]) / np.sum(samples.shape[0])
                        if params['sampling_strategy'] == 'adaptive':
                            violations = self._epoch_sample_buffer[i]['buffer']['violations']
                            p = violations / np.sum(violations)
                        self._epoch_sample_idc[i] = np.random.choice(samples.shape[0], p=p, replace=False,
                                                                     size=con.max_pool_size // num_batches)
                        con.u = samples[self._epoch_sample_idc[i], :]

                # slice batches
                b_x, b_y = t_x[batch * batch_size: (batch + 1) * batch_size, :], \
                    t_y[batch * batch_size: (batch + 1) * batch_size, :]

                # closure to recompute loss for optimizers like lbfgs
                def closure():
                    if torch.is_grad_enabled():
                        optimizer.zero_grad()
                    loss = self.__loss(self.layers(b_x), b_y, params)
                    if loss.requires_grad:
                        loss.backward(retain_graph=True)
                    return loss
                optimizer.step(closure)

                # compute loss one final time to get proper cout / metrics from __loss (maybe remove this for performance)
                self._epoch_cout_buffer = ''
                loss = self.__loss(self.layers(b_x), b_y, params)

            # calculate error on validation data
            loss_str = f"Epoch {epoch + 1}/{params['max_epoch']} [{time() - start:.3f}s] - train: "
            if do_validation:
                self.__loss(self.layers(t_x), t_y, params)
                train_loss = self._epoch_metrics_buffer['crit']

                self._epoch_cout_buffer = ''
                loss = self.__loss(self.layers(t_x_val), t_y_val, params)
                self._epoch_metrics_buffer['crit_train'] = train_loss
                self._epoch_metrics_buffer['crit_val'] = self._epoch_metrics_buffer['crit']
                self._epoch_metrics_buffer.pop('crit')
                loss_str += f"{train_loss:.6f} [{self._criterion.__class__.__name__}], validation: "

            # without val data use the train data for threshold test
            elif batch_size != x.shape[0]:
                self._epoch_cout_buffer = ''
                loss = self.__loss(self.layers(t_x), t_y, params)

            self._epoch_cout_buffer = loss_str + self._epoch_cout_buffer
            self.__stdout(verbosity=CMLP.CMD_LEVEL_IMPORTANT)
            history.append(copy.deepcopy(self._epoch_metrics_buffer))

            # abort training if loss fell below 'threshold'
            loss = loss.detach().numpy()
            if loss < params['threshold']:
                self.__stdout(f"{loss:.3f}(Loss) fell below {self.threshold}, learning is aborted")
                break

            if is_pressed('esc'):
                self.__stdout("Learning was interrupted manually")
                break

        self._epoch_cout_buffer = ''
        self._epoch_metrics_buffer = {}
        return self.__reformat_history(history)

    def __objective_function(self, x):
        self.__last_layer_activation(x).detach().numpy()

    def __resample_constraints(self, params):
        self._epoch_sample_buffer = [self.__verify_constraint(con) for con in self._continuous_constraints]

        for i, con in enumerate(self._continuous_constraints):
            data = self._epoch_sample_buffer[i]
            violations = data['buffer']['violations']
            samples = data['buffer']['samples']

            # update sample pool
            if params['sampling_strategy'] == 'max' or violations.shape[0] < con.samples_per_iteration:
                sort_idc = np.argsort(violations)[::-1]
                con.u = utils.stack([con.u, samples[sort_idc[:con.samples_per_iteration]]], 'v')
            elif params['sampling_strategy'] == 'weighted':
                p = np.exp(violations*self._lambda)
                if any(np.isinf(p)):
                    p = violations
                    print('vioations too large for softmax')
                idc = np.random.choice(samples.shape[0], p=p/np.sum(p),
                                       replace=False, size=con.samples_per_iteration)

                con.u = utils.stack([con.u, samples[idc]], 'v')

            # trim sample pool if it gets too large
            if con.u.shape[0] > con.max_pool_size:
                violations = self.__calculate_violation(con, con.u, mask_negative_values=False)
                con.u = con.u[np.argsort(violations)[::-1], :]

            # console print
            metrics = data['metrics']
            if self._previous_epoch_sample_buffer:  # first epoch print is different
                prev_metrics = self._previous_epoch_sample_buffer[i]['metrics']
                self._epoch_cout_buffer += 'Resample for ' + con.label + f' ({len(con.u)}) [{100*metrics["reliability"]:.2f}% ({(metrics["reliability"]-prev_metrics["reliability"]) * 100:+.2f}), '\
                                                                         f'{metrics["mean"]:.4f} ({metrics["mean"]-prev_metrics["mean"]:+.4f}), '\
                                                                         f'{metrics["max"]:.4f} ({metrics["max"]-prev_metrics["max"]:+.4f})]\n'
            else:
                self._epoch_cout_buffer += 'Resample for ' + con.label + f' [{100*metrics["reliability"]:.2f}%, {metrics["mean"]:.2f}, {metrics["max"]:.2f}]\n'

    def __loss(self, y_pred, y, params):
        # reconstruction
        total = self._criterion(y_pred, y)
        crit = copy.deepcopy(total.detach().numpy())
        cout = f"{crit:.6f} [{self._criterion.__class__.__name__}]"
        self._epoch_metrics_buffer['crit'] = crit

        # constraint penalty
        if params['constrained']:
            penalty = self.__constraint_penalty()
            total += self._lambda * penalty
            cout += f", {penalty.detach().numpy():.6f} [Penalty]"
            self._epoch_metrics_buffer['penalty'] = penalty.detach().numpy()

        # L2 regularization
        if self._reg is not None:
            reg = self.__l2()
            total += self._reg * reg
            cout += f", {reg.detach().numpy():.6f} [L2]"
            self._epoch_metrics_buffer['L2'] = reg.detach().numpy()

        self._epoch_cout_buffer = cout + self._epoch_cout_buffer
        self._epoch_metrics_buffer['total'] = total.detach().numpy()
        self._epoch_metrics_buffer['lambda'] = self._lambda
        return total

    def __stdout(self, s=None, verbosity=2, exact_verbosity=None):
        if s is None:
            s = self._epoch_cout_buffer
            self._epoch_cout_buffer = ''
        if exact_verbosity is not None:
            if self._verbose == exact_verbosity:
                print(s)
        else:
            if verbosity <= self._verbose:
                print(s)

    def __jac(self, t_u):
        if self._normalize:
            t_inp_scale = torch.FloatTensor(self._inp_scale).to(self.device)[:, 0]
        jac = jacobian(lambda x: self.layers(x).sum(0), t_u, vectorize=True, create_graph=True).transpose(0, 1)
        jac = jac / t_inp_scale[None, None, :]
        return jac

    def __constraint_penalty(self):
        penalty = torch.tensor(0., requires_grad=True)
                
        # calculate loss per constraint
        self._epoch_metrics_buffer['constraint_metrics'] = [{'mean': 0., 'max': 0., 'label': con.label} for con in self._constraints]
        for i, con in enumerate(self._constraints):
            samples = self.__check_sample_dim(con.u)

            if samples.size == 0:
                continue

            viol = self.__calculate_violation(con, samples, to_numpy=False)
            viol *= con.weight / self._total_constraint_weight

            # add positive squared violation to loss
            penalty = penalty + torch.mean(viol**2)

            # track metrics
            max_viol = torch.max(viol).detach().numpy()
            mean_viol = torch.mean(viol).detach().numpy()

            # update violations for adaptive sampling
            for j, idx in enumerate(self._epoch_sample_idc[i]):
                self._epoch_sample_buffer[i]['buffer']['violations'][idx] = viol[j].detach().numpy()

            if self._verbose >= 2:
                self._epoch_cout_buffer += f'\n     | {con.label}({len(con.u)}): ' \
                                           f'{mean_viol:.3f} [mean], ' \
                                           f'{max_viol:.3f} [max]'
            self._epoch_metrics_buffer['constraint_metrics'][i] = {'mean': mean_viol, 'max': max_viol, 'label': con.label}
        return penalty

    def __calculate_violation(self, con, samples, to_numpy=True, mask_negative_values=True, use_test_bounds=False):
        if self._normalize:
            samples_trf = self.normalize(samples)
        else:
            samples_trf = samples

        # calculate constraint value (the whole functional without bounds)
        if isinstance(con, LinearConstraint):
            value = 0
            for idx, pd in enumerate(con.partials):
                value += self.__fct(samples_trf, pd, con.factors[idx], to_numpy=False)  # we only work with tensors
        else:
            value = torch.squeeze(con.func(self, samples))

        # calculate bounds and violation
        if use_test_bounds:
            ub = self.__get_bound(con.max_test_value, samples_trf, to_numpy=False)
            lb = self.__get_bound(con.min_test_value, samples_trf, to_numpy=False)
        else:
            if isinstance(con, InequalityConstraint):
                ub = self.__get_bound(con.max_value, samples_trf, to_numpy=False)
                lb = self.__get_bound(con.min_value, samples_trf, to_numpy=False)
            else:
                ub = self.__get_bound(con.value, samples_trf, to_numpy=False)
                lb = self.__get_bound(con.value, samples_trf, to_numpy=False)
        if all(torch.isinf(ub)):
            viol = -(value - lb)
        elif all(torch.isneginf(lb)):
            viol = value - ub
        else:
            viol = torch.max(value - ub, -(value - lb))

        if mask_negative_values:
            viol[viol <= 0] = 0

        if to_numpy:
            return viol.detach().numpy()
        else:
            return viol

    def __reformat_history(self, hist):
        return {key: [entry[key] for entry in hist] for key in hist[0].keys()}

    def __get_optimizer(self, params=None):
        # get optimizer from string, parameters are either default values or received from the 'optim_params' dictionary
        if params is None:
            params = self.__validate_train_params()
        optimizer_name = params['optimizer']
        optimizer_params = params['optim_params']
        try:
            optim_params = {k.lower(): v for k, v in optimizer_params.items()}  # all keys to lower case
            optim_dict = {
                "adadelta": optim.Adadelta(self.layers.parameters(),
                                           lr=params['lr'] if params['lr'] is not None else 1.0,
                                           rho=optim_params.get("rho", 0.9),
                                           eps=optim_params.get("eps", 1e-6),
                                           weight_decay=optim_params.get("weight_decay", 0)),
                "adagrad": optim.Adagrad(self.layers.parameters(),
                                         lr=params['lr'] if params['lr'] is not None else 0.01,
                                         lr_decay=optim_params.get("lr_decay", 0),
                                         eps=optim_params.get("eps", 1e-10),
                                         initial_accumulator_value=optim_params.get("initial_accumulator_value", 0),
                                         weight_decay=optim_params.get("weight_decay", 0)),
                "adam": optim.Adam(self.layers.parameters(),
                                   lr=params['lr'] if params['lr'] is not None else 0.001,
                                   betas=optim_params.get("betas", (0.9, 0.999)),
                                   eps=optim_params.get("eps", 1e-08),
                                   amsgrad=optim_params.get("amsgrad", False),
                                   weight_decay=optim_params.get("weight_decay", 0)),
                "adamw": optim.AdamW(self.layers.parameters(),
                                     lr=params['lr'] if params['lr'] is not None else 0.001,
                                     betas=optim_params.get("betas", (0.9, 0.999)),
                                     eps=optim_params.get("eps", 1e-08),
                                     amsgrad=optim_params.get("amsgrad", False),
                                     weight_decay=optim_params.get("weight_decay", 0)),
                "sparseadam": optim.SparseAdam(self.layers.parameters(),
                                               lr=params['lr'] if params['lr'] is not None else 0.001,
                                               betas=optim_params.get("betas", (0.9, 0.999)),
                                               eps=optim_params.get("eps", 1e-08)),
                "adamax": optim.Adamax(self.layers.parameters(),
                                       lr=params['lr'] if params['lr'] is not None else 0.002,
                                       betas=optim_params.get("betas", (0.9, 0.999)),
                                       eps=optim_params.get("eps", 1e-08),
                                       weight_decay=optim_params.get("weight_decay", 0)),
                "asgd": optim.ASGD(self.layers.parameters(),
                                   lr=params['lr'] if params['lr'] is not None else 0.01,
                                   lambd=optim_params.get("lambd", 0.0001),
                                   alpha=optim_params.get("alpha", 0.75),
                                   t0=optim_params.get("t0", 1e6),
                                   weight_decay=optim_params.get("weight_decay", 0)),
                "lbfgs": optim.LBFGS(self.layers.parameters(),
                                     lr=params['lr'] if params['lr'] is not None else 1,
                                     max_iter=optim_params.get("max_iter", 6),
                                     max_eval=optim_params.get("max_eval", None),
                                     tolerance_grad=optim_params.get("tolerance_grad", 1e-07),
                                     tolerance_change=optim_params.get("tolerance_change", 1e-09),
                                     history_size=optim_params.get("history_size", int(params['resample_freq'])-1),
                                     line_search_fn=optim_params.get("line_search_fn", None)),
                "rmsprop": optim.RMSprop(self.layers.parameters(),
                                         lr=params['lr'] if params['lr'] is not None else 0.01,
                                         alpha=optim_params.get("alpha", 0.99),
                                         eps=optim_params.get("eps", 1e-08),
                                         momentum=optim_params.get("momentum", 0),
                                         centered=optim_params.get("centered", False),
                                         weight_decay=optim_params.get("weight_decay", 0)),
                "sgd": optim.SGD(self.layers.parameters(),
                                 lr=params['lr'] if params['lr'] is not None else 0.001,
                                 dampening=optim_params.get("dampening", 0),
                                 momentum=optim_params.get("momentum", 0),
                                 nesterov=optim_params.get("nesterov", False),
                                 weight_decay=optim_params.get("weight_decay", 0))
            }
            optimizer_name = optimizer_name.lower()
            if optimizer_name not in optim_dict:
                raise ValueError(f"'{optimizer_name}' is not a valid optimizer")
            optimizer = optim_dict.get(optimizer_name)
        except:
            raise

        return optimizer

    def __l2(self):
        weight_sum = torch.tensor(0., requires_grad=True)
        for name, param in self.layers.named_parameters():
            if 'weight' in name:
                weight_sum = weight_sum + param.pow(2.0).sum()
        return weight_sum

    def __verify_constraint(self, con):
        samples = self.__check_sample_dim(con.draw_test_samples())

        violations = self.__calculate_violation(con, samples, mask_negative_values=False, use_test_bounds=True)
        msk = violations > 0

        if np.sum(msk) == 0:  # no violation
            metrics = {'reliability': 1,  'mean': 0, 'max': 0}
            buffer = {'samples': np.array([[]]), 'violations':  np.array([])}
        else:
            pos_violations = violations[msk]
            pos_samples = samples[msk, :]

            metrics = {'reliability': 1 - (np.sum(msk) / con.test_samples_per_iteration),
                       'mean': np.mean(pos_violations), 'max': np.max(pos_violations)}
            buffer = {'samples': pos_samples, 'violations': pos_violations}
        return {'metrics': metrics, 'buffer': buffer}

    def __last_layer_activation(self, x):
        """Calculates the activations in the last layer"""

        # turn input into valid tensor
        if not torch.is_tensor(x):
            if not isinstance(x, np.ndarray):
                raise ValueError("'x' must be a numpy array or tensor")
            x = torch.from_numpy(x).float()
        x.requires_grad = True

        # get all child modules
        layers = self.layers[:-1] if isinstance(self.layers[-1], nn.Linear) else self.layers[:-2]  # strip last linear layer
        stripped_net = nn.Sequential(OrderedDict(
            (f"layer_{i}", layer) for (i, layer) in enumerate(layers)))  # create temporal mlp from remaining modules
        return stripped_net(x)

    def __clear_constraints(self):
        """empties the matrices that define the constraints during the QP"""
        self._aeq = np.array([])
        self._beq = np.array([])
        self._aieq = np.array([])
        self._bieq = np.array([])

    def __fill_ieq_matrix(self):
        """fill matrices that define the inequality constraints during the QP

        Generally the resulting matrix are of the form

            (out_dim * hid_dim) x (2 * Number of samples from all IEQ-constraints)

        but can have fewer rows when max-/min_values are inf/-inf, as those
        constraints are redundant. This shape allows the inclusion of tikhonov
        regularization.
        """
        for con in self._eq_constraints:
            # skip invalid or redundant constraints
            if (con.max_value == np.inf and con.min_value == -np.inf) or con.max_value == -np.inf or con.min_value == np.inf:
                continue
            if not isinstance(con, LinearConstraint):
                continue

            # reshape samples if necessary and skip empty constraints
            u = self.__check_sample_dim(np.asarray(con.u))
            if u.size == 0:
                continue
            if self._normalize:
                u = self.normalize(u)
            ub = self.__get_bound(con.max_value, u)
            lb = self.__get_bound(con.min_value, u)

            num_samples = u.shape[0]
            partials = con.partials
            coeffs = con.factors

            for idx, pd in enumerate(partials):
                AieqMinTemp = np.array([])
                bieqMinTemp = np.zeros((num_samples,))
                AieqMaxTemp = np.array([])
                bieqMaxTemp = np.zeros((num_samples,))

                factors = coeffs[idx]
                if callable(factors):
                    factors = factors(u)

                for s in range(self.out_dim):

                    # compute sub-row for each sample in con.u
                    hid_derivative = np.array([])
                    for inp in u:
                        hid_derivative_temp = np.array([])
                        inp = torch.from_numpy(inp).float()
                        inp.requires_grad = True
                        hid_state = self.__last_layer_activation(inp)
                        for h in range(self.layers[-1].in_features):  # utils.stack partial derivatives horizontally
                            hid_derivative_temp = utils.stack((hid_derivative_temp,
                                                         self.__compute_derivative(hid_state[h],
                                                                                   inp,
                                                                                   pd[s]).detach().numpy()), 'h')

                        if len(pd[s]) == 0:  # if no derivative was taken for this row, the bias remains 1
                            hid_derivative_temp = utils.stack((hid_derivative_temp, np.array([1])), 'h')
                        else:
                            utils.stack((hid_derivative_temp, np.array([0])), 'h')
                        hid_derivative = utils.stack((hid_derivative, hid_derivative_temp))

                    # join sub-rows horizontally and skip redundant rows
                    if con.min_value != -np.inf:
                        AieqMinTemp = utils.stack((AieqMinTemp, -np.tile(factors[s], (self.layers[-1].in_features + 1, 1)).T * hid_derivative), 'h')
                        bieqMinTemp -= lb
                    if con.max_value != np.Inf:
                        AieqMaxTemp = utils.stack((AieqMaxTemp, np.tile(factors[s], (self.layers[-1].in_features + 1, 1)).T * hid_derivative), 'h')
                        bieqMaxTemp += ub
                bieqMinTemp /= self.out_dim
                bieqMaxTemp /= self.out_dim
                if idx == 0:
                    AieqTempSum = utils.stack((AieqMinTemp, AieqMaxTemp))
                    if con.max_value != np.Inf and con.min_value == -np.inf:
                        bieqTempSum = bieqMaxTemp
                    elif con.max_value == np.Inf and con.min_value != -np.inf:
                        bieqTempSum = bieqMinTemp
                    else:
                        bieqTempSum = utils.stack((bieqMinTemp, bieqMaxTemp), 'h')
                else:
                    AieqTempSum += utils.stack((AieqMinTemp, AieqMaxTemp))
                    if con.max_value != np.Inf and con.min_value == -np.inf:
                        bieqTempSum += bieqMaxTemp
                    elif con.max_value == np.Inf and con.min_value != -np.inf:
                        bieqTempSum += bieqMinTemp
                    else:
                        bieqTempSum += utils.stack((bieqMinTemp, bieqMaxTemp), 'h')

            self._aieq = np.reshape(utils.stack((self._aieq, AieqTempSum)), (-1, (self.layers[-1].in_features + 1) * self.out_dim))
            self._bieq = utils.stack([self._bieq, np.array([bieqTempSum]).flatten()], 'h')
            self._bieq = self._bieq.flatten()

            if self._aieq.ndim == 1:
                self._aieq = np.array([self._aieq])

    def __fill_eq_matrix(self):
        """fill matrices that define the equality constraints during the QP

        Generally the resulting matrix are of the form

            (out_dim * hid_dim) x (Number of samples from all EQ-constraints)

        This shape allows the inclusion of tikhonov regularization.
        """
        for c, con in enumerate(self._eq_constraints):
            if not isinstance(con, LinearConstraint):
                continue

            u = self.__check_sample_dim(np.asarray(con.u))
            if u.size == 0:
                continue
            if self._normalize:
                u = self.normalize(u)
            tar = self.__get_bound(con.value, u)

            partials = con.partials
            coeffs = con.factors

            if u.size == 0:
                continue

            for idx, pd in enumerate(partials):
                AeqTemp = np.array([])
                beqTemp = 0

                factors = coeffs[idx]
                if callable(factors):
                    factors = factors(u)

                for s in range(self.out_dim):
                    # compute sub-row for each sample in u
                    hid_derivative = np.array([])
                    for inp in u:
                        hid_derivative_temp = np.array([])
                        inp = torch.from_numpy(inp).float()
                        inp.requires_grad = True
                        hid_state = self.__last_layer_activation(inp)
                        for h in range(self.layers[-1].in_features):
                            hid_derivative_temp = utils.stack((hid_derivative_temp,
                                                         self.__compute_derivative(hid_state[h],
                                                                                   inp,
                                                                                   pd[s]).detach().numpy()), 'h')

                        if len(pd[s]) == 0:  # if no derivative was taken for this row, the bias remains 1
                            hid_derivative_temp = utils.stack((hid_derivative_temp, np.array([1])), 'h')
                        else:
                            utils.stack((hid_derivative_temp, np.array([0])), 'h')
                        hid_derivative = utils.stack((hid_derivative, hid_derivative_temp))

                    # join sub-rows horizontally
                    AeqTemp = utils.stack((AeqTemp, -np.tile(factors[s], (self.layers[-1].in_features + 1, 1)).T * hid_derivative), 'h')
                    beqTemp -= tar

                if idx == 0:
                    AeqTempSum = AeqTemp
                    beqTempSum = beqTemp
                else:
                    AeqTempSum = AeqTempSum + AeqTemp
                    beqTempSum = beqTempSum + beqTemp

            self._aeq = utils.stack((self._aeq, AeqTempSum))
            self._beq = np.append(self._beq, beqTempSum / self.out_dim)

            if self._aeq.ndim == 1:
                self._aeq = np.array([self._aeq])

    def __fct(self, x, partials, factors, to_numpy=True):
        """Calculate linear combination of partial derivatives of the hidden state."""

        # reshape input if necessary
        x = np.asarray(x) if not utils.check_array(x) else x
        if x.ndim == 1:
            num_samples = 1
            x = np.array([x])
        else:
            num_samples = x.shape[0]

        if callable(factors):
            factors = torch.FloatTensor(factors(x)).to(self.device)

        # compute value for each output node individually
        out_dim = len(partials)
        val = torch.zeros((num_samples, out_dim))
        for s in range(out_dim):
            hid = torch.zeros((x.shape[0], ))
            if partials[s] == [[[]]]:  # for constraints w/ partial derivatives, input can be processed batch-wise
                inp = torch.from_numpy(x).float()
                inp.requires_grad = True
                hid = self.layers(inp)[:, s]
            else:
                for i, inp in enumerate(x):  # autograd.grad only accepts scalar tensors [2021]
                    inp = torch.from_numpy(inp).float()
                    inp.requires_grad = True
                    hid[i] = self.__compute_derivative(self.layers(inp)[s], inp, partials[s])
            val[:, s] = hid * factors[s]
        out = torch.sum(val, dim=1)  # return sum over all output nodes
        if to_numpy:
            out = out.detach().numpy()
        return out

    def __compute_derivative(self, out, inp, wrt):
        """Computes arbitraty partial derivatives of out w.r.t to inp.

        Args:
            out (torch.tensor): value for which the partial derivative is computed, must be a scalar or 1-dimensional.
            inp (torch.tensor): value from which the deriative is taken, must be a scalar or 1-dimensional.
            D (list): list of ints, that determine, w.r.t to which input node the derivative is taken. Length of
                'wrt' determines the degree of differentiation, an empty list returns the function value, 'out'.

        Returns:
            The partial derivative at that point.

        Raises:
            ValueError: If 'inp' or 'out' are not tensors
        """
        # check if 'inp' and 'out' are tensors
        if not torch.is_tensor(out) or not torch.is_tensor(inp):
            raise ValueError("'inp' and 'out' must be tensors")

        if self._normalize:
            t_inp_scale = torch.FloatTensor(self._inp_scale).to(self.device)

        # repeatedly take derivative and reassign the output
        order = len(wrt)
        for n in range(order):
            out = grad(out, inp, create_graph=True, retain_graph=True)[0][wrt[n]]
            if self._normalize:
                out = out / t_inp_scale[wrt[n]]
        return out

    def __qp(self, x, y, analytical=True):
        """Solves the linearly constrained quadratic optimization problem.

        Solves a problem of the form

            min { 1/2 * ||h*x-d||^2 }
            subject to Aeq*x = beq, A*x <= b

        In this case specificially, x corresponds to the output weigths, where as
        h corresponds to the last layer activation, therefore minimizing the
        MSE of target values and model output. Furthermore h, x and d are reshaped:

            h: num_samples x hid_dim -->  (num_samples * out_dim) x (hid_dim * out_dim)
            d: num_samples x out_dim -->  (num_samples * out_dim) x 1
            x: hid_dim x out_dim --> (hid_dim * out_dim) x 1

        h is transformed by creating a matrix of size (num_samples * out_dim) x (hid_dim * out_dim)
        filled with zeros and adding the original h on the diagonal. This ensures that
        the resulting matrix multiplication is identical to the original one. This
        transformation enables the use of tikhonov regularization.
        The used solvers minimizes a more general class of problems that can be formulated as
        follows:

            min { 1/2 * x.T * P * x + q.T * x}
            subject to Aeq*x = beq, A*x <= b

        where x.T denotes the vector transpose of x. h and d can also be applied to this problem:

            P = h.T * T
            q = -h.T * d

        The solution for x is then reshaped back into its original shape.
        """
        d = y.flatten('F')
        h = self.__objective_function(x)
        h = utils.stack((h, np.ones((h.shape[0], 1))), 'h')

        # reshape h and d to add tikhonov regularization
        xH = h.shape[0]
        yH = h.shape[1]
        for i in range(self.out_dim):
            if i == 0:
                C = np.hstack([np.zeros((xH, i * yH)), h, np.zeros((xH, (self.out_dim - i - 1) * yH))])
            else:
                C = np.vstack(
                    [C, np.hstack([np.zeros((xH, i * yH)), h, np.zeros((xH, (self.out_dim - i - 1) * yH))])])
        C = np.vstack([C, np.sqrt(self._tik) * np.eye((self.layers[-1].in_features + 1) * self.out_dim)])
        d = np.hstack([np.array([d]), np.zeros((1, (self.layers[-1].in_features + 1) * self.out_dim))]).T

        # trf constraint matrices
        no_eqs = (self._aeq.size == 0)
        no_ieqs = (self._aieq.size == 0)
        if not no_eqs:
            inv = 1 / self._beq
            mask = (inv == np.inf) | (inv == -np.inf)
            Aeq = copy.deepcopy(self._aeq)
            beq = copy.deepcopy(self._beq)
            is_negative = beq < 0
            Aeq[~mask] /= np.reshape(beq[~mask], (-1, 1))
            beq[~mask] = 1.
            Aeq[is_negative] *= -1.
            beq[is_negative] *= -1.
        if not no_ieqs:
            inv = 1 / self._bieq
            mask = (inv == np.inf) | (inv == -np.inf)
            Aieq = copy.deepcopy(self._aieq)
            bieq = copy.deepcopy(self._bieq)
            is_negative = bieq < 0
            Aieq[~mask] /= np.reshape(bieq[~mask], (-1, 1))
            bieq[~mask] = 1.
            Aieq[is_negative] *= -1.
            bieq[is_negative] *= -1.

        if not analytical:
            if not no_eqs:
                compensation = x.shape[0] / Aeq.shape[0]
                C = np.vstack([C, compensation * self._lambda * Aeq])
                d = np.vstack([d, compensation * self._lambda * beq[:, np.newaxis]])
            if self._aieq.size != 0:
                # because we encode ieq constraints as eq constraints, only active ones are included in the minimization
                out_weights = np.reshape(np.hstack([self.layers[-1].weight.detach().numpy(), np.atleast_2d(self.layers[-1].bias.detach().numpy()).T]), (1, -1))
                active_constraint_msk = out_weights @ Aieq.T > bieq

                compensation = x.shape[0] / active_constraint_msk.shape[0]
                C = np.vstack([C, compensation * self._lambda * Aieq[active_constraint_msk[0], :]])
                d = np.vstack([d, compensation * self._lambda * bieq[active_constraint_msk[0], np.newaxis]])
            self.__clear_constraints()
            no_eqs = True
            no_ieqs = True

        # transform C and d so that the solver can accept them
        P = matrix(C.T.dot(C))
        q = matrix(-1 * C.T.dot(d))

        # solve QP
        if no_eqs and no_ieqs:
            sol = solvers.qp(P, q)
        elif no_eqs:
            Aieq = matrix(Aieq.astype(np.double))
            bieq = matrix(bieq.astype(np.double))
            sol = solvers.qp(P, q, G=Aieq, h=bieq)
        elif no_ieqs:
            Aeq = matrix(Aeq.astype(np.double))
            beq = matrix(beq.astype(np.double))
            sol = solvers.qp(P, q, A=Aeq, b=beq)
        else:
            Aieq = matrix(Aieq.astype(np.double))
            bieq = matrix(bieq.astype(np.double))
            Aeq = matrix(Aeq.astype(np.double))
            beq = matrix(beq.astype(np.double))
            sol = solvers.qp(P, q, G=Aieq, h=bieq, A=Aeq, b=beq)#, kktsolver="ldl")

        # reshape solution and set output weights
        terminated = sol['status'] != 'unknown'
        if terminated:
            with torch.no_grad():
                solution = np.reshape(np.asarray(sol['x']), (self.layers[-1].in_features + 1, self.out_dim), order='F')
                out_weights = torch.from_numpy(solution[:self.layers[-1].in_features].T).float()
                bias = torch.from_numpy(np.reshape(solution[self.layers[-1].in_features:], (self.out_dim,), order='F')).float()
                out_weights.requires_grad = True
                bias.requires_grad = True
                list(self.layers.parameters())[-2].data = out_weights
                list(self.layers.parameters())[-1].data = bias

                inputs = torch.from_numpy(x).float()
                targets = torch.from_numpy(y).float()
                inputs.requires_grad = True
                outputs = self.layers(inputs)
                loss = self._criterion(outputs, targets).detach().numpy()
            self._epoch_metrics_buffer['crit'] = loss
            solver_msg = f"{str(self._criterion.__class__.__name__)[:-2]}: {loss:.5f}, Solution [{sol['status']} - loss : penalty, 1 : {self._lambda: .2f}]\n"


        else:
            self._epoch_metrics_buffer['crit'] = np.nan
            solver_msg = "Solver did not terminate. Abort constrained learning.\n"

        return solver_msg, terminated

    def __get_eq_test_value(self, value, eps=0):
        if callable(value):
            return lambda x: value(x) + eps
        else:
            return value + eps

    def __check_sample_dim(self, u):
        if u.ndim == 1:
            return np.reshape(u, (1, u.shape[0]))
        else:
            return u

    def __get_bound(self, bound, u, to_numpy=True):
        if callable(bound):
            if self._normalize:
                u = self.denormalize(u)
            bound = np.reshape(bound(u), (-1,))
        else:
            bound = np.repeat(bound, u.shape[0])
        if to_numpy:
            return bound
        else:
            return torch.from_numpy(bound).to(self.device)

    def __validate_train_params(self, **kwargs):
        if not self._initialized:
            raise ValueError("Network must be initialled with a dataset (X, Y) or a generator, yielding (X, Y)'s")

        params = {
            'constrained': True,
            'sampling_strategy': 'weighted',
            'max_epoch': 1000,
            'lambda_max': 3,
            'resample_freq': 10,
            'optimizer': 'lbfgs',
            'optim_params': {},
            'threshold': -np.inf,
            'batch_size': None,
            'lr': None,
            'validation_ratio': None,          
        }
        
        # todo: finish training kw validation
        #if 'batch_size' in kwargs and kwargs['batch_size'] is not None and (not isinstance(kwargs['batch_size'], int) or kwargs['batch_size'] < 1):
        #    raise ValueError("'batch_size' must either be None or a positive integer")
        #if 'max_epoch' in kwargs and not isinstance(kwargs['max_epoch'], int) or kwargs['max_epoch'] < 1:
        #    raise ValueError("'max_epoch' must be a positive integer")
        #if 'lr' in kwargs and kwargs['lr'] is not None and (not isinstance(kwargs['lr'], float) or kwargs['lr'] <= 0):
        #    raise ValueError("'lr' must either be None or a positive float")
        #if 'threshold' in kwargs and kwargs['threshold'] is not None and (not isinstance(kwargs['threshold'], float) or kwargs['threshold'] <= 0):
        #    raise ValueError("'threshold' must either be None or a positive float")

        params.update(kwargs)
        return params


class TwoLayerCMLP(CMLP):

    def __init__(self, inp_dim: int, out_dim: int,
                 hid_dim: int = 30,
                 max_iter=1000,
                 reg: float = 1e-6,
                 mu: float = 0.3,
                 eps: float = 1e-3,
                 bip: bool = False,
                 normalize: bool = False,
                 verbose: int = 2):

        if not isinstance(inp_dim, int) or inp_dim < 1:
            raise ValueError("'inp_dim' must be of type int and greater than 0")
        if not isinstance(out_dim, int) or out_dim < 1:
            raise ValueError("'out_dim' must be of type int and greater than 0")
        if not isinstance(bip, bool):
            raise ValueError("'bip' must be of type bool")
        if not isinstance(eps, (float, int)) or eps <= 0:
            raise ValueError("'eps' must be of type float and greater than 0")
        if not isinstance(reg, (float, int)) or reg <= 0:
            raise ValueError("'reg' must be of type float and greater than 0")
        if not isinstance(mu, (float, int)):
            raise ValueError("'mu' must be of type float")
        if not isinstance(normalize, bool):
            raise ValueError("'normalize' must be of type bool")
        if not isinstance(verbose, int) or verbose not in (0, 1, 2):
            raise ValueError("'verbose' must be of type int and between 0 and 2")
        if not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError("'max_iter' must be of type int and greater than 0")

        CMLP.__init__(inp_dim, out_dim, loss=nn.MSELoss(), verbose=verbose,
                      eps=eps, tik=tik, reg=reg, normalize=normalize)

        self.hid_dim = hid_dim
        self.add_layer(torch.nn.Linear(inp_dim, hid_dim))
        self.add_layer(torch.nn.Linear(hid_dim, out_dim))
        self.max_iter = max_iter
        self._obj_fcts = []

    def train(self, skip_initial_qp=False,
              analytical=False,  # constraints are enforced through penalty term instead of hard constraints
              constraint_weight=1,  # weight of constraints if analytical is True
              constraint_weight_factor=1,  # by how much constraint_weight is increased each iterationy
              ieq_test_epsilon=None):
        self.sqp(self, max_iter=self.max_iter,
                 skip_initial_qp=skip_initial_qp,
                 analytical=analytical,
                 constraint_weight=constraint_weight,
                 constraint_weight_factor=constraint_weight_factor,
                 ieq_test_epsilon=ieq_test_epsilon)

    def add_objective_function(self, *args: ObjFct):
        for obj_fct in args:
            if not isinstance(obj_fct, ObjFct):
                raise ValueError('Each list element must be a ObjFct!')
            obj_fct.inp_dim = self.inp_dim
            self._obj_fcts.append(obj_fct)

    def __objective_function(self, x):
        """Fill hidden state matrix according to list of linear objective functions"""
        # todo: rewrite this in terms of a general MLP

        if not isinstance(x, np.ndarray):
            raise ValueError("'x' must be a numpy array")

        if not self._obj_fcts:
            h = self._ELM__calc_hidden_state(x)
            return (utils.stack((h, np.ones((h.shape[0], 1))), 'h'), y)

        input_weights = self.input_weights.T
        a = self.a.T
        b = self.b.T

        Heff = np.array([])
        for of, objfct in enumerate(self._obj_fcts):
            num_samples = x.shape[0]
            if num_samples == 0:
                return

            partials = objfct.partials
            coeffs = objfct.factors

            for idx, pd in enumerate(partials):
                HTemp = np.array([])
                factors = coeffs[idx]
                factors = coeffs[idx]
                if callable(factors):
                    factors = factors(u)

                for s in range(self.out_dim):
                    # compute sub-row for each sample in con.u
                    hid_derivative = np.array([])
                    for inp in u:
                        hid_derivative_temp = np.array([])
                        inp = torch.from_numpy(inp).float()
                        inp.requires_grad = True
                        hid_state = self.__last_layer_activation(inp)
                        for h in range(self.layers[-1].in_features):  # utils.stack partial derivatives horizontally
                            hid_derivative_temp = utils.stack((hid_derivative_temp,
                                                         self.__compute_derivative(hid_state[h],
                                                                                   inp,
                                                                                   pd[s]).detach().numpy()), 'h')

                        if len(pd[s]) == 0:  # if no derivative was taken for this row, the bias remains 1
                            hid_derivative_temp = utils.stack((hid_derivative_temp, np.array([1])), 'h')
                        else:
                            hid_derivative_temp = utils.stack((hid_derivative_temp, np.array([0])), 'h')
                        hid_derivative = utils.stack((hid_derivative, hid_derivative_temp))
                    HTemp = utils.stack((HTemp, np.tile(factors[s], (self.hid_dim + 1, 1)).T * hid_derivative), 'h')

                if idx == 0:
                    h = HTemp
                else:
                    h += HTemp

            Heff = utils.stack((Heff, h))
        return Heff






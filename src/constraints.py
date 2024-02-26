__version__ = '3.1'
__author__ = 'Jannick Stranghöner'

from abc import ABC
from typing import List, Callable, Generator, Optional

import numpy as np

from regions import SamplingRegion

"""
This module contains a set of classes to define different kinds of constraints. 
Custom constraints can easily be defined by subclassing (...)Constraint classes, 
for reference take a look at the (D/C)(I)EQC definition. The BaseConstraint constructor
invokes all other parent class constructors with the appropriate named arguments, so
you do not have to repeat all the keyword arguments from the, possibly many, parent classes.
Sadly there is no way of signaling this behavior to the  static analysis, so you lose
constructor auto completion (at least I have not found a way to do this in PyCharm / 
VSCode).
"""


class BaseConstraint(ABC):

    def __init__(self, weight: float = 1.0, label: str = '', lagrange_alpha: float = 1e-3, **kwargs):
        self.inp_dim = None
        self.weight = weight
        self.label = label
        self.lagrange_alpha = lagrange_alpha

        # a constraint has many super constructors by design, and each gets called
        # with respective named arguments by this method
        self.initialize_child_class_from_kwargs(**kwargs)

    def initialize_child_class_from_kwargs(self, **kwargs):
        for cls in self.__class__.__bases__:
            if cls is BaseConstraint:
                continue

            # call each super constructor with respective arguments
            kwargs_sub = {}
            for var_name in cls.__init__.__code__.co_varnames[1:cls.__init__.__code__.co_argcount]:
                if var_name in kwargs:
                    kwargs_sub[var_name] = kwargs[var_name]
            cls.__init__(self, **kwargs_sub)


class DiscreteConstraint(ABC):
    def __init__(self, u):
        self.u = u


class ContinuousConstraint(ABC):
    def __init__(self, region: SamplingRegion,
                 samples_per_iteration: int = 1,
                 test_samples_per_iteration: int = 1000,
                 satisfaction_threshold: float = 0.998,
                 max_pool_size: int = 25,
                 max_test_value: Optional[float] = None,
                 min_test_value: Optional[float] = None):
        self.region = region
        self.samples_per_iteration = samples_per_iteration
        self.test_samples_per_iteration = test_samples_per_iteration
        self.satisfaction_threshold = satisfaction_threshold
        self.max_pool_size = max_pool_size
        self.max_test_value = max_test_value
        self.min_test_value = min_test_value
        self.u = np.array([[]])

    def draw_test_samples(self, sample_size: Optional[int] = None):
        if sample_size is None:
            sample_size = self.test_samples_per_iteration
        return self.region.sample(sample_size)


class InequalityConstraint(ABC):
    def __init__(self, max_value: float, min_value: Optional[float] = -np.inf):
        self.min_value = min_value
        self.max_value = max_value


class EqualityConstraint(ABC):
    def __init__(self, value: float, eps: Optional[float] = 1e-3):
        self.value = value
        self.eps = eps


class LinearConstraint(ABC):
    def __init__(self, partials: List, factors: List):
        """Value to be constrained is a linear combination of arbitrary partial derivatives.

        The syntax of the input parameters is straight-forward, and best explained using examples:

        given a CELM f that realizes a R3 -> R2 function (f_(n) corresponds f_(n)(x) for readability)
        partials=[[[], []]], factors=[[1, 0]]       corresponds to C = 1 * f_0 + 0 * f_1
        partials=[[[1], [2]]], factors=[[0.5, 0.5]] corresponds to C = 0.5 * df_0 / dx_1 + 0.5 * df_1 / dx_2
        partials=[[[], []], factors=[g]             corresponds to C = g(x)_0 * f_0 + g(x)_1 * f_1
        partials=[[[], []], [[0,1], []]], factors=[g, [2, 0]]
                                                    corresponds to C = g(x)_0 * f_0 + g(x)_1 * f_1 +
                                                                       2 * df_0^2 / (dx_0 * d_x1)

        """
        self.partials = partials
        self.factors = factors


class NonlinearConstraint(ABC):
    def __init__(self, func: Callable):
        self.func = func


class CIEQC(BaseConstraint, LinearConstraint, ContinuousConstraint, InequalityConstraint):
    def __init__(self, *args, **kwargs):
        BaseConstraint.__init__(self, *args, **kwargs)


class CEQC(BaseConstraint, LinearConstraint, ContinuousConstraint, EqualityConstraint):
    def __init__(self, *args, **kwargs):
        BaseConstraint.__init__(self, *args, **kwargs)


class DIEQC(BaseConstraint, LinearConstraint, DiscreteConstraint, InequalityConstraint):
    def __init__(self, *args, **kwargs):
        BaseConstraint.__init__(self, *args, **kwargs)


class DEQC(BaseConstraint, LinearConstraint, DiscreteConstraint, EqualityConstraint):
    def __init__(self, *args, **kwargs):
        BaseConstraint.__init__(self, *args, **kwargs)

class ObjFct:
    def __init__(self, partials, factors, inp_dim=1):
        self.partials = partials
        self.factors = factors
        self.inp_dim = inp_dim


if __name__ == "__main__":
    con = DIEQC(2, partials=[[[0]]], factors=[[1]], u=[[0]])
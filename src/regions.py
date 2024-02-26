__all__ = ['Box', 'BoxSurface', 'ParametricSurface']
__version__ = '3.1'
__author__ = 'Jannick Stranghöner'

from abc import ABC, abstractmethod
import numpy as np

import utils


class SamplingRegion(ABC):
    @abstractmethod
    def get_mass(self):
        pass

    @abstractmethod
    def sample(self, num_samples):
        pass


class Box(SamplingRegion):
    """N-dimensional box whose sides are orthogonal to the coordinate axes."""
    def __init__(self, upper_bounds, lower_bounds):

        self.upper_bounds = upper_bounds if utils.check_array(upper_bounds) else np.asarray(upper_bounds)
        self.lower_bounds = lower_bounds if utils.check_array(lower_bounds) else np.asarray(lower_bounds)

        if self.upper_bounds.shape != self.lower_bounds.shape or self.upper_bounds.ndim != 1:
            raise ValueError("The dimensionality of minimum and maximum values does not match")
        else:
            self.inp_dim = self.upper_bounds.shape[0]

    def __contains__(self, item):
        for i in range(self.upper_bounds.shape[0]):
            if not (self.lower_bounds[i] <= item[i] <= self.upper_bounds[i]):
                return False
        return True

    def get_mass(self):
        return np.prod(np.abs(self.upper_bounds - self.lower_bounds))

    def sample(self, num_samples):
        return np.random.uniform(self.lower_bounds, self.upper_bounds, size=(num_samples, self.inp_dim))


class BoxSurface(Box):
    """Surface of N-dimensional box whose sides are orthogonal to the coordinate axes."""
    def __init__(self, upper_bounds, lower_bounds):
        super().__init__(upper_bounds, lower_bounds)

        self.surfaces = [None for _ in range(self.inp_dim * 2)]

        for i in range(self.inp_dim):
            mask = np.ones((self.inp_dim)).astype(bool)
            mask[i] = 0
            self.surfaces[2 * i] = OrthogonalSurface(upper_bounds=self.upper_bounds[mask],
                                           lower_bounds=self.lower_bounds[mask],
                                           offset=self.upper_bounds[i],
                                           offset_dim=i,
                                           side='top')
            self.surfaces[2 * i + 1] = OrthogonalSurface(upper_bounds=self.upper_bounds[mask],
                                               lower_bounds=self.lower_bounds[mask],
                                               offset=self.lower_bounds[i],
                                               offset_dim=i,
                                               side='bottom')

    def get_mass(self):
        return sum([s.get_mass() for s in self.surfaces])

    def sample(self, num_samples):
        samples = np.zeros((num_samples, self.inp_dim))

        # each side gets #points proportional to their mass
        prev_idx = 0
        total_mass = self.get_mass()
        for i, s in enumerate(self.surfaces):
            if i == len(self.surfaces)-1:
                _num_samples = num_samples - prev_idx
            else:
                _num_samples = int(s.get_mass() / total_mass * num_samples)
            samples[prev_idx: prev_idx + _num_samples] = s.sample(_num_samples)
            prev_idx += _num_samples
        return samples

    def get_normal_func(self):
        def normal(X):
            normals = np.zeros(X.shape)

            for i, x in enumerate(X):
                surface = self.surfaces[np.argmin([s.distance(x) for s in self.surfaces])]
                normals[i] = surface.normal(x)
            return normals

        return normal


class OrthogonalSurface(SamplingRegion):
    """(N-1)-dimensional box surface that sits orthogonal on the 'offset_dim' axis at 'offset'."""
    def __init__(self, upper_bounds,
                 lower_bounds,
                 offset,
                 offset_dim,
                 side):  # in ['top', 'bottom'], used for surface normal direction
        self.inp_dim = upper_bounds.shape[0] + 1
        mask = np.ones(self.inp_dim).astype(bool)
        mask[offset_dim] = 0

        self._upper_bounds = upper_bounds
        self._lower_bounds = lower_bounds

        self.upper_bounds = np.zeros(self.inp_dim)
        self.lower_bounds = np.zeros(self.inp_dim)

        self.upper_bounds[mask] = upper_bounds
        self.lower_bounds[mask] = lower_bounds
        self.upper_bounds[offset_dim] = offset
        self.lower_bounds[offset_dim] = offset

        self._surface = Box(self.upper_bounds, self.lower_bounds)  # surface as dim(upper_bounds)-1 dimensional box
        self.offset = offset
        self.offset_dim = offset_dim
        self.side = side

    def get_mass(self):
        msk = np.ones(self.inp_dim).astype(bool)
        msk[self.offset_dim] = 0
        return np.prod(np.abs(self.upper_bounds[msk] - self.lower_bounds[msk]))

    def sample(self, num_samples):
        return self._surface.sample(num_samples)

    def distance(self, point):
        return np.abs(point[self.offset_dim] - self.offset)

    def normal(self, point):
        normal = np.zeros(self.inp_dim)
        normal[self.offset_dim] = 1 if self.side == 'top' else -1
        return normal


class ParametricSurface(SamplingRegion):
    def __init__(self, funcs, t_min=0.0, t_max=1.0):
        self.t_max = t_max
        self.t_min = t_min
        self.funcs = funcs

    def get_mass(self):
        # todo: integrate this
        return 0

    def sample(self, num_samples):
        t = np.random.uniform(self.t_min, self.t_max, (num_samples,))
        return np.array([f(t) for f in self.funcs]).T

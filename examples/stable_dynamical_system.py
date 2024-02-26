import sys
import os
sys.path.append(os.path.abspath('./src'))

from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import dill as pickle
from sklearn.preprocessing import MinMaxScaler
import pyLasaDataset as lasa

from src.regions import Box
from src.constraints import DEQC, CIEQC, ObjFct
from src.learner import CELM
import src.utils as utils

# build dataset
demo_0 = lasa.DataSet.Angle.demos[2]
x = demo_0.pos.T
y = demo_0.vel.T

scaler = MinMaxScaler()
x = scaler.fit_transform(x)
x -= x[-1, :]
y *= scaler.scale_

# bounds
x_max = np.max(x, axis=0)
x_min = np.min(x, axis=0)
x_range = x_max - x_min
x_max += 0.3 * x_range
x_min -= 0.3 * x_range
region = Box(np.reshape(x_max, (-1,)), np.reshape(x_min, (-1,)))

### train lyapunov candidate from data ###
obj_fcts = [
    ObjFct(partials=[[[0]]], factors=[[-1]], inp_dim=2),
    ObjFct(partials=[[[1]]], factors=[[-1]], inp_dim=2),
]  # the gradient of the scalar lyapunov candidate should follow the data

cieqcs = [
    CIEQC(region=region, partials=[[[]]], factors=[[1]], max_value=np.Inf, min_value=0),
    CIEQC(region=region, partials=[[[0]], [[1]]], max_value=np.inf, min_value=0,
          factors=[lambda x: np.array([x[:, 0]]), lambda x: np.array([x[:, 1]])])
]  # ensures positive definiteness of the lyapunov candidate

candidate = CELM(inp_dim=2,
                 out_dim=1,
                 hid_dim=50,
                 deqcs=[
                     DEQC(partials=[[[]]], factors=[[1]], value=0, u=[0, 0]),
                     DEQC(partials=[[[0]]], factors=[[1]], value=0, u=[0, 0]),
                     DEQC(partials=[[[1]]], factors=[[1]], value=0, u=[0, 0])  # first and second lyapunov condition
                 ],
                 cieqcs=cieqcs,
                 obj_fcts=obj_fcts)

print('Train lyapunov candidate ...')
candidate.init(x)
candidate.train(x, y)

# plot candidate function
num_points = 10000
xx, yy = np.meshgrid(np.linspace(region.lower_bounds[0], region.upper_bounds[0], int(np.sqrt(num_points)))[::-1],
                     np.linspace(region.lower_bounds[1], region.upper_bounds[1], int(np.sqrt(num_points)))[::-1])
inp_seq = utils.make_seq_from_grid(xx, yy)
out_seq = candidate.apply(inp_seq)
zz = utils.make_grid_from_seq(out_seq[:, 0].T, int(np.sqrt(num_points)), int(np.sqrt(num_points)))


### learn dynamical system ###

# callable factor for scalar product in constraint
def lyapunov_gradient_factor(x):
    grad = candidate.grad(x).T
    return grad


model = CELM(inp_dim=2,
             out_dim=2,
             hid_dim=50,
             deqcs=[
                 DEQC(partials=[[[], []]], factors=[[1, 0]], value=0, u=np.array([0, 0])),
                 DEQC(partials=[[[], []]], factors=[[0, 1]], value=0, u=np.array([0, 0]))
             ],  # sets equilibrium point to (0,0))
             cieqcs=[
                 CIEQC(region=region,
                       partials=[[[], []]],
                       factors=[lyapunov_gradient_factor],
                       satisfaction_threshold=0.998,
                       max_value=np.inf,
                       min_value=0)
             ]  # forces stability through lyapunov candidate
             )

print('\nTrain dynamical system ...')
model.init(x)
model.train(x, y)
p, _ = utils.integrate(model, x0=x[0], order=1, t_max=20, dt=0.001,
                       ub=region.upper_bounds, lb=region.lower_bounds, method='single-step')

### plot results ###
plt.figure(1).suptitle('Lyapunov Candidate', fontsize=20)
plt.xlabel('x')
plt.ylabel('v')
plt.contour(xx, yy, zz, levels=100)
plt.colorbar()
plt.scatter(x[:, 0], x[:, 1], label='Demonstration')
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.legend()

plt.figure(2).suptitle('Reproduction (Position)', fontsize=20)
plt.plot(x[:, 0], x[:, 1], color='blue', label='Dataset')
plt.scatter(0, 0, s=100, marker='D', color='r', label='Target')
plt.scatter(p[0, 0], p[0, 1], s=100, marker='*', color='r', label='Starting point')
plt.plot(p[:, 0], p[:, 1], '-', color='r', label='Reproduction')

# quiver
out_seq = model.apply(inp_seq)
uu = utils.make_grid_from_seq(out_seq[:, 0].T, int(np.sqrt(num_points)), int(np.sqrt(num_points)))
vv = utils.make_grid_from_seq(out_seq[:, 1].T, int(np.sqrt(num_points)), int(np.sqrt(num_points)))
norm = np.sqrt(uu ** 2 + vv ** 2)
plt.quiver(xx, yy, uu / norm, vv / norm, angles='uv')

plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.legend()
plt.show()

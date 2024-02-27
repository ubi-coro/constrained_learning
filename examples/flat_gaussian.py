import sys
import os
sys.path.append(os.path.abspath('./src'))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src.regions import Box
from src.learner import CELM
from src.constraints import CIEQC
from src.utils import make_seq_from_grid, make_grid_from_seq

# Parameters for the 2D Gaussian
mean = np.array([0, 0])
covariance = np.array([[1, 0], [0, 1]])

# Compute 2D Gaussian distribution on (x,y) grid
xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
xy = make_seq_from_grid(xx, yy)
z = np.exp(-0.5 * np.sum(np.dot(xy, np.linalg.inv(covariance)) * xy, axis=1))
z /= 2 * np.pi * np.sqrt(np.linalg.det(covariance))

# Define the model and constraints
region = Box(lower_bounds=[-3, -3], upper_bounds=[3, 3])
con = CIEQC(region=region, max_value=0.1, partials=[[[]]], factors=[[1]])
model = CELM(inp_dim=2,
             out_dim=1,
             hid_dim=30,
             cieqcs=[con])

# Fit the gaussian
model.init(xy)
model.train(xy, z)
z_hat = model.apply(xy)
z_u = model.apply(con.u)

# Plotting
zz = make_grid_from_seq(z, 100, 100)
zz_hat = make_grid_from_seq(z_hat, 100, 100)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xx, yy, zz, cmap='Greys', edgecolor='k', alpha=.3)
ax.plot_surface(xx, yy, zz_hat, cmap='Reds', edgecolor='r', alpha=.3)
ax.scatter(con.u[:, 0], con.u[:, 1], z_u, color='b', s=70, label='Constraint Samples')

# empty handles for legend
plt.plot([], [], color='k', label='Ground Truth')
plt.plot([], [], color='r', label='Constrained Approximation [f(x,y) < 0.1]')
ax.set_title('Surface Plot of 2D Gaussian Distribution')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f(x, y)')
plt.legend()
plt.show()
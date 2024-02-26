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

# quadratic
x = np.linspace(-3, 3)
y = 2*x**2 - 3

# Define the model and constraints
region = Box(lower_bounds=[-3], upper_bounds=[3])
con = CIEQC(region=region,
            max_value=np.inf,
            min_value=0,
            partials=[[[0]]],
            factors=[[1]])  # constrain first derivative
model = CELM(inp_dim=1,
             out_dim=1,
             hid_dim=30,
             cieqcs=[con])

# Fit the gaussian
model.init(x)
model.train(x, y)
y_hat = model.apply(x)

# Plotting
plt.figure(figsize=(10, 7))
plt.plot(x, y, color='k', label='Ground Truth')
plt.plot(x, y_hat, color='r', label="Constrained Approximation [f'(x) > 0]")
plt.scatter(con.u, model.apply(con.u), color='b', label='Constraint Samples')
plt.gca().set_title('Monotone Quadratic')
plt.gca().set_xlabel('x')
plt.gca().set_ylabel('f(x)')
plt.legend()
plt.show()
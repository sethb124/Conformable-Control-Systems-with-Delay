from sys import argv

import matplotlib.pyplot as plt
import numpy as np
from numpy.ma import cos, exp


def z(t):
    return 5 * exp(-0.09 * t) * cos(0.6 * t)


# some parameters
omega = 0.8
delta = 0.1

# setting up the system
A = np.matrix([[0, 1], [-omega ** 2, -2 * delta * omega]])
B = np.c_[[0, 1]]
C = np.matrix('1 0')
q = 0.001
p = 3
tf = 50
dt = 0.001
t_dis = np.arange(0, tf + dt, dt)
Sf = p * C.T @ C
vf = C.T * p * z(tf)

# these are the conformable derivatives we want
alphas = [0.34, 0.67, 1]

# set up plotting
plt.rcParams["font.size"] = 8
fig, axes = plt.subplots(len(alphas), 2)
fig.tight_layout(rect=(0, 0, 1, 0.95), pad=1.5)

# loop over rows of plot and alphas
for axis, alpha in zip(axes, alphas):
    # S is an empty array of matrices
    # with last element given by Sf
    S = np.empty((t_dis.size, 2, 2))
    S[-1] = Sf

    # loop backward and do Euler's method to find S
    # this solves the Ricatti equation
    for i in range(t_dis.size - 1, 0, -1):
        dSdt = -(dt * (i + 1)) ** (alpha - 1) * \
            (A.T @ S[i] + S[i] @ A - S[i] @ B @ B.T @ S[i] + q * C.T @ C)
        S[i - 1] = S[i] - dt * dSdt

    # v is an output equation or something idk
    # it's needed to solve for stuff
    # in example 4.1, it was 0
    v = np.empty((t_dis.size, 2, 1))
    v[-1] = vf

    for i in range(t_dis.size - 1, 0, -1):
        dvdt = -(dt * (i + 1)) ** (alpha - 1) * \
            ((A - B @ B.T @ S[i]).T @ v[i] + q * C.T * z(dt * (i + 1)))
        v[i - 1] = v[i] - dt * dvdt

    # x is an array of column vectors
    x = np.empty((t_dis.size, 2, 1))
    x[0] = np.c_[[10, 3]]
    # u is an array of real #s
    u = np.empty(t_dis.size)

    # loop forward and solve state space equation
    # this also uses Euler's method as far as I can tell
    for i in range(0, t_dis.size - 1):
        u[i] = -(B.T @ S[i] @ x[i] + B.T @ v[i]).item()
        x[i + 1] = x[i] + (dt * (i + 2)) ** (alpha - 1) * \
            dt * (A @ x[i] + B * u[i])
    u[-1] = -(B.T @ S[-1] @ x[-1] + B.T @ v[-1]).item()

    # plot u (1st column)
    axis[0].plot(t_dis, u)
    axis[0].set_title(f'$\\alpha = {alpha}$')
    axis[0].margins(x=0)

    # plot y = x1 and z (2nd column)
    axis[1].plot(t_dis, x[:, 0], label='Position [m]')
    axis[1].plot(t_dis, z(t_dis), label='Desired Position [m]', color='gold')
    axis[1].legend()
    axis[1].margins(x=0)

# you can use -s to save the plot to a file
# generally you don't want to do this
if len(argv) == 2 and argv[1] == '-s':
    plt.savefig(argv[0][:-3] + '.png')
else:
    plt.show()

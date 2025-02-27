from sys import argv

import matplotlib.pyplot as plt
import numpy as np
from numpy.ma import cos, sin


def z1(t):
    return np.piecewise(t, [0 <= t < 1, 1 <= t < 2, 2 <= t <= 3],
                        [lambda t: cos(2 * np.pi * t),
                         lambda t: t ** 2 * (1 - t) / 2,
                         lambda t: cos(4 * np.pi * t) / 2 + 1])


def z2(t):
    return np.piecewise(t, [0 <= t < 1, 1 <= t < 2, 2 <= t <= 3],
                        [lambda t: 1.2 * t ** 2 * (1 - t),
                         lambda t: cos(2 * np.pi * t),
                         lambda t: sin(4 * np.pi * t) / 5 - 0.5])


def z(t):
    return np.c_[[z1(t), z2(t)]]


# setting up the system
A = np.matrix('-9  4   4.5 -2;'
              '-3  0.4 0.7 -6;'
              ' 5  0.3 5    3;'
              ' 4 -2.5 2    3', dtype=np.longdouble)
B = np.matrix(' 1    1.5 0  ;'
              ' 0.3  2   0.4;'
              ' 0.3 -0.3 0  ;'
              '-0.3 -1   0.5', dtype=np.longdouble)
C = np.matrix(' 1 0 2  1;'
              '-1 1 0 -1', dtype=np.longdouble)
Q = 10 ** 4 * np.identity(2)
tf = 3
dt = 0.0005
ts = np.arange(0, tf + dt, dt)
Sf = np.zeros((4, 4))
vf = np.zeros((4, 1))

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
    S = np.empty((ts.size, 4, 4), dtype=np.longdouble)
    S[-1] = Sf

    # loop backward and do Euler's method to find S
    # this solves the Ricatti equation
    for i in range(ts.size - 1, 0, -1):
        dSdt = -(dt * (i + 1)) ** (alpha - 1) * \
            (A.T @ S[i] + S[i] @ (A - B @ B.T @ S[i]) + C.T @ Q @ C)
        S[i - 1] = S[i] - dt * dSdt

    # v is an output equation or something idk
    # it's needed to solve for stuff
    # in example 4.1, it was 0
    v = np.empty((ts.size, 4, 1), dtype=np.longdouble)
    v[-1] = vf

    for i in range(ts.size - 1, 0, -1):
        dvdt = -(dt * (i + 1)) ** (alpha - 1) * \
            ((A - B @ B.T @ S[i]).T @ v[i] + C.T @ Q @ z(dt * (i + 1)))
        v[i - 1] = v[i] - dt * dvdt

    # x is an array of column vectors
    x = np.empty((ts.size, 4, 1), dtype=np.longdouble)
    x[0] = np.c_[[-0.25, -0.5, 0.25, -0.3]]
    # u is an array of real #s
    u = np.empty((ts.size, 3, 1), dtype=np.longdouble)

    # loop forward and solve state space equation
    # this also uses Euler's method as far as I can tell
    for i in range(0, ts.size - 1):
        u[i] = -B.T @ S[i] @ x[i] + B.T @ v[i]
        x[i + 1] = x[i] + (dt * (i + 2)) ** (alpha - 1) * \
            dt * (A @ x[i] + B @ u[i])
    u[-1] = -B.T @ S[-1] @ x[-1] + B.T @ v[-1]

    # plot u1 and u2 (1st column)
    axis[0].plot(ts, u[:, 0], label='$u_1(t)$')
    axis[0].plot(ts, u[:, 1], label='$u_2(t)$', color='gold')
    axis[0].plot(ts, u[:, 2], label='$u_3(t)$', color='green')
    axis[0].set_title(f'$\\alpha = {alpha}$')
    axis[0].legend()
    axis[0].margins(x=0)

    # plot y1 and y2 (2nd column)
    y = C @ x
    axis[1].plot(ts, y[:, 0], label='$y_1(t)$')
    axis[1].plot(ts, y[:, 1], label='$y_2(t)$', color='gold')
    axis[1].legend()
    axis[1].margins(x=0)

# axes[0, 0].set_ylim(-10, 5.5)
axes[0, 0].set_ylim(-15, 15)
axes[0, 1].set_ylim(-2, 2)
axes[1, 0].set_ylim(-15, 15)
axes[2, 0].set_ylim(-15, 15)
# you can use -s to save the plot to a file
# generally you don't want to do this
if len(argv) == 2 and argv[1] == '-s':
    plt.savefig(argv[0][:-3] + '.png')
else:
    plt.show()

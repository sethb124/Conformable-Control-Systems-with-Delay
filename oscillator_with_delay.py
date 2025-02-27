# do the damped harmonic oscillator problem again
# but with the delay introduced
# do a delay of 10

from sys import argv

import matplotlib.pyplot as plt
import numpy as np

# some parameters
omega = 0.8
delta = 0.1

# setting up the system
A = np.matrix([[0, 1], [-omega ** 2, -2 * delta * omega]])
B = np.c_[[0, 1]]
Q = 0.001 * np.identity(2)
# Q = np.identity(2)
tf = 20
dt = 0.001
ts = np.arange(0, tf + dt, dt)
delay = int(10 / dt)
Sf = np.matrix('3 0; 0 1')

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
    S = np.empty((ts.size + delay, 2, 2))
    S[ts.size - 1] = Sf
    for i in range(ts.size, ts.size + delay):
        S[i] = np.identity(2)
    for i in range(0, delay):
        S[i] = -(dt * i) ** alpha / alpha * Q

    # loop backward and do Euler's method to find S
    # this solves the Ricatti equation for t >= t_0 + h
    for i in range(ts.size - 1, delay, -1):
        dSdt = -(dt * (i + 1)) ** (alpha - 1) * \
            (A.T @ S[i] + S[i] @ A - S[i] @ B @ B.T @ S[i - delay] + Q)
        S[i - 1] = S[i] - dt * dSdt
    # I now basically have initial conditions, so
    # I could theoretically go forward and solve,
    # but this gives a vastly different solution
    # and kinda ignores the final condition
    # for i in range(delay - 1, ts.size - 1):
    #     dSdt = -(dt * (i - 1)) ** (alpha - 1) * \
    #         (A.T @ S[i - delay] + S[i - delay] @ A - S[i] @ B @ B.T @ S[i] + Q)
    #     S[i + 1] = S[i] + dt * dSdt

    # x is an array of column vectors
    x = np.empty((ts.size + delay, 2, 1))
    x[0] = np.c_[[10, 3]]
    # u is an array of real #s
    u = np.empty(ts.size + delay)
    # this tries to solve homogeneous equation for t < 0
    # the problem is that the conformable derivative is
    # not defined for values of t < 0
    # for i in range(-1, -delay - 1, -1):
    #     x[i] = x[i + 1] - (dt * (i + 1)) ** (alpha - 1) * dt * A @ x[i + 1]
    #     u[i] = -(B.T @ S[i] @ x[i]).item()
    for i in range(ts.size, ts.size + delay):
        x[i] = np.c_[[10, 3]]
        u[i] = -(B.T @ S[i] @ x[i]).item()

    # loop forward and solve state space equation
    # this also uses Euler's method as far as I can tell
    for i in range(0, ts.size - 1):
        u[i] = -(B.T @ S[i] @ x[i]).item()
        x[i + 1] = x[i] + (dt * (i + 2)) ** (alpha - 1) * \
            dt * (A @ x[i - delay] + B * u[i - delay])
    u[ts.size - 1] = -(B.T @ S[ts.size - 1] @ x[ts.size - 1]).item()

    # plot u (1st column)
    axis[0].plot(ts, u[0:ts.size])
    axis[0].set_title(f'$\\alpha = {alpha}$')
    axis[0].margins(x=0)

    # plot x1 and x2 (2nd column)
    axis[1].plot(ts, x[:ts.size, 0], label='Position [m]')
    axis[1].plot(ts, x[:ts.size, 1], label='Velocity [m/s]', color='gold')
    axis[1].legend()
    axis[1].margins(x=0)

# you can use -s to save the plot to a file
# generally you don't want to do this
if len(argv) == 2 and argv[1] == '-s':
    plt.savefig(argv[0][:-3] + '.png')
else:
    plt.show()

import math as math
from sys import argv

import matplotlib.pyplot as plt
import numpy as np

# def S(t):
#     return (1 - sym.exp(sym.sqrt(404) * (0.5 - t))) * (10 + sym.sqrt(101) -
#                                                        sym.exp(sym.sqrt(404) * (0.5 - t)) / (10 - sym.sqrt(101)))
#

# setting up the system
tf = 0.5
dt = 0.001
ts = np.arange(0, tf + dt, dt)
delay = int(0.25 / dt)

# set up plotting
plt.rcParams["font.size"] = 8
fig, axes = plt.subplots(2, 1)
fig.tight_layout(rect=(0, 0, 1, 0.95), pad=1.5)

# S = np.empty(ts.size + delay)
S = np.empty(ts.size)


# # loop backward and do Euler's method to find S
# # this solves the Ricatti equation
# for i in range(ts.size - 1, 0, -1):
#     dSdt = -(dt * (i + 1)) ** (alpha - 1) * \
#         (A.T @ S[i] + S[i] @ A - S[i] @ B @ B.T @ S[i] + Q)
#     S[i - 1] = S[i] - dt * dSdt
# S[ts.size - 1] = 0
S[-1] = 0
for i in range(ts.size - 1, delay, -1):
    S[i - 1] = S[i] - dt * (1 - 20 * S[i] - S[i] * S[i])
for i in range(delay, 0, -1):
    S[i - 1] = S[i] - dt * (1 - S[i] * S[i])

# axes[0].plot(ts, S)

# for i in range(-delay, ts.size):
#     S[i] = (1 - math.exp(math.sqrt(404) * (0.5 - i * dt))) * (10 + math.sqrt(101) -
#                                                               math.exp(math.sqrt(404) * (0.5 - i * dt)) / (10 - math.sqrt(101)))
#

x = np.empty(ts.size)
x[0] = 1

u = np.empty(ts.size)

for i in range(0, delay):
    x[i + 1] = x[i] + dt * (10 + S[i - delay])

for i in range(delay, ts.size - 1):
    x[i + 1] = x[i] + dt * (10 + S[i - delay]) * x[i - delay]
    # u[i + 1] = S[i - delay] * x[i - delay]

axes[0].plot(ts, x)
axes[1].plot(ts, S * x)
#
# u = S[:ts.size] * x
# # axes[0].plot(np.arange(-0.25, 0, dt), S[ts.size:])
# # axes[0].plot(ts, S[:ts.size])
# axes[0].plot(ts, x)
# axes[0].plot(ts[:delay], x[:delay])
# axes[1].plot(ts[delay:], x[delay:])
# axes[1].plot(ts, u)
# axes[0].plot(ts[:delay], u[:delay])
# axes[1].plot(ts[delay:], u[delay:])


# # x is an array of column vectors
# x = np.empty((ts.size, 2, 1))
# x[0] = np.c_[[10, 3]]
# # u is an array of real #s
# u = np.empty(ts.size)
#
# # loop forward and solve state space equation
# # this also uses Euler's method as far as I can tell
# for i in range(0, ts.size - 1):
#     u[i] = -(B.T @ S[i] @ x[i]).item()
#     x[i + 1] = x[i] + (dt * (i + 2)) ** (alpha - 1) * \
#         dt * (A @ x[i] + B * u[i])
# u[-1] = -(B.T @ S[-1] @ x[-1]).item()
#
# # plot u (1st column)
# axis[0].plot(ts, u)
# axis[0].set_title(f'$\\alpha = {alpha}$')
# axis[0].margins(x=0)
#
# # plot x1 and x2 (2nd column)
# axis[1].plot(ts, x[:, 0], label='Position [m]')
# axis[1].plot(ts, x[:, 1], label='Velocity [m/s]', color='gold')
# axis[1].legend()
# axis[1].margins(x=0)

# you can use -s to save the plot to a file
# generally you don't want to do this
if len(argv) == 2 and argv[1] == '-s':
    plt.savefig(argv[0][:-3] + '.png')
else:
    plt.show()

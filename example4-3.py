import numpy as np
from numpy.ma import sin, cos
import matplotlib.pyplot as plt
from sys import argv

def z(t):
	return np.c_[[sin(t), cos(t)]]

# setting up the system
A = np.matrix(' 0.809 -2.060  0.325  0.465  0.895;' \
			  ' 6.667  0.200  1.333  0      0.667;' \
			  '-1.291  0.458 -1.072 -2.326 -0.199;' \
			  '-0.324  0.824  1.670 -1.186 -0.358;' \
			  '-3.509 -4.316 -0.702  0     -8.351')
B = np.matrix(' 0.955 -0.379;' \
			  '-1.667 -1.667;' \
			  '-0.212  1.195;' \
			  ' 0.618  0.052;' \
			  ' 0.877  1.403')
C = np.matrix('2 0   1 0   0;' \
			  '0 1.5 0 1.2 1')
Q = 10 ** 3 * np.identity(2)
tf = 6
dt = 0.001
t_dis = np.arange(0, tf + dt, dt)
Sf = np.zeros((5, 5))
vf = np.zeros((5, 1))

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
	S = np.empty((t_dis.size, 5, 5))
	S[-1] = Sf

	# loop backward and do Euler's method to find S
	# this solves the Ricatti equation
	for i in range(t_dis.size - 1, 0, -1):
		dSdt = -(dt * (i + 1)) ** (alpha - 1) * (A.T @ S[i] + S[i] @ (A - B @ B.T @ S[i]) + C.T @ Q @ C)
		S[i - 1] = S[i] - dt * dSdt;

	# v is an output equation or something idk
	# it's needed to solve for stuff
	# in example 4.1, it was 0
	v = np.empty((t_dis.size, 5, 1))
	v[-1] = vf

	for i in range(t_dis.size - 1, 0, -1):
		dvdt = -(dt * (i + 1)) ** (alpha - 1) * ((A - B @ B.T @ S[i]).T @ v[i] + C.T @ Q @ z(dt * (i + 1)))
		v[i - 1] = v[i] - dt * dvdt

	# x is an array of column vectors
	x = np.empty((t_dis.size, 5, 1))
	x[0] = np.c_[[0.05, 0.05, 0.05, 0.05, 0.05]]
	# u is an array of real #s
	u = np.empty((t_dis.size, 2, 1))

	# loop forward and solve state space equation
	# this also uses Euler's method as far as I can tell
	for i in range(0, t_dis.size - 1):
		u[i] = -B.T @ S[i] @ x[i] + B.T @ v[i]
		x[i + 1] = x[i] + (dt * (i + 2)) ** (alpha - 1) * dt * (A @ x[i] + B @ u[i])
	u[-1] = -B.T @ S[-1] @ x[-1] + B.T @ v[-1]

	# plot u1 and u2 (1st column)
	axis[0].plot(t_dis, u[:,0], label='$u_1(t)$')
	axis[0].plot(t_dis, u[:,1], label='$u_2(t)$', color='gold')
	axis[0].set_title(f'$\\alpha = {alpha}$')
	axis[0].legend()
	axis[0].margins(x=0)

	# plot y1 and y2 (2nd column)
	y = C @ x
	axis[1].plot(t_dis, y[:,0], label='$y_1(t)$')
	axis[1].plot(t_dis, y[:,1], label='$y_2(t)$', color='gold')
	axis[1].legend()
	axis[1].margins(x=0)

axes[0, 0].set_ylim(-10, 5.5)
axes[0, 1].set_ylim(-1.2, 1.2)
axes[1, 0].set_ylim(-10, 5.5)
axes[2, 0].set_ylim(-10, 5.5)
# you can use -s to save the plot to a file
# generally you don't want to do this
if len(argv) == 2 and argv[1] == '-s':
	plt.savefig(argv[0][:-3] + '.png')
else:
	plt.show()

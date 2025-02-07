import numpy as np
import matplotlib.pyplot as plt

# some parameters
omega = 0.8
delta = 0.1

# setting up the system
A = np.matrix([[0, 1], [-omega ** 2, -2 * delta * omega]])
B = np.c_[[0, 1]]
Sf = np.matrix('3 0; 0 1')
Q = 0.001 * np.identity(2)
R = 1
tf = 20
dt = 0.001
t_dis = np.arange(0, tf + dt, dt)

# these are the conformable derivatives we want
alphas = [0.34, 0.66, 1]

# set up plotting
plt.rcParams["font.size"] = 8
fig, axes = plt.subplots(len(alphas), 2)
fig.tight_layout(rect=(0, 0, 1, 0.95), pad=1.5)

# loop over rows of plot and alphas
for axis, alpha in zip(axes, alphas):
	# S is an empty array of 2x2 matrices
	# with last element given by Sf
	# not sure where Sf comes from (ask Wintz)
	S = np.empty((t_dis.size, 2, 2))
	S[-1] = Sf

	# loop backward and do Euler's method to find S
	# this solves the Ricatti equation
	for i in range(t_dis.size - 1, 0, -1):
		dSdt = -(dt * (i + 1)) ** (alpha - 1) * (A.T @ S[i] + S[i] @ A - S[i] @ B @ B.T @ S[i] + Q)
		S[i - 1] = S[i] - dt * dSdt;

	# x is an array of 2x1 column vectors
	x = np.empty((t_dis.size, 2, 1))
	x[0] = np.c_[[10, 3]]
	# u is an array of real #s
	u = np.empty(t_dis.size)

	# loop forward and solve state space equation
	# this also uses Euler's method as far as I can tell
	for i in range(0, t_dis.size - 1):
		u[i] = -(B.T @ S[i] @ x[i]).item()
		x[i + 1] = x[i] + (dt * (i + 2)) ** (alpha - 1) * dt * (A @ x[i] + B * u[i])
	u[-1] = -(B.T @ S[-1] @ x[-1]).item()

	# plot u (1st column)
	axis[0].plot(t_dis, u)
	axis[0].set_title(f'$\\alpha = {alpha}$')
	axis[0].margins(x=0)

	# plot x and x' (2nd column)
	axis[1].plot(t_dis, x[:,0], label='Position [m]')
	axis[1].plot(t_dis, x[:,1], label='Velocity [m/s]', color='gold')
	axis[1].legend()
	axis[1].margins(x=0)

# pick exactly one of these and comment out the other one
# generally you don't want to save the plot locally
plt.show()
# plt.savefig('figure1.png')

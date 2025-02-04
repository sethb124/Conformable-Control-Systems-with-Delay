import numpy as np
import matplotlib.pyplot as plt
from jitcdde import jitcdde, y, t
import control as ct

alphas = [1, 0.66, 0.33]
# alphas = np.linspace(1, 0, 3)
systems = [[
	t ** (alpha - 1) * y(1, t),
	t ** (alpha - 1) * (-0.64 * y(0, t) - 0.16 * y(1, t))]
for alpha in alphas]

plt.rcParams['font.size'] = 8
fig, axs = plt.subplots(len(alphas), 1)
fig.tight_layout(rect=(0, 0, 1, 0.95), pad=3.0)

for i, (system, alpha) in enumerate(zip(systems, alphas)):
	dde = jitcdde(system, max_delay=100.)
	ts = np.linspace(0, 20, 2000)
	dde.constant_past([10, 3]) #, -0.05])
	dde.integrate_blindly(0.001)
	ys = []
	for t in ts:
		ys.append(dde.integrate(t))
	ys=np.array(ys)
	axs[i].plot(ts, ys[:,0], color='blue', linewidth=1, label='Position [m]')
	axs[i].plot(ts, ys[:,1], color='gold', linewidth=1, label='Velocity [m/s]')
	axs[i].set_title('$\\alpha=$' + str(alpha))
	axs[i].legend()
	axs[i].margins(0, 0.1)

# the control package may allow us to find
# u(t), but I don't think it'd be able to
# once a delay is introduced
A = [[0, 1], [-0.64, -0.16]]
B = [[0], [1]]
Q = [[0.001, 0], [0, 0.001]]
R = 1

K, _, _ = ct.lqr(A, B, Q, R)


plt.show()

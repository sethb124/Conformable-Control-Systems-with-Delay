import numpy as np
import matplotlib.pyplot as plt
from jitcdde import jitcdde, y, t

plt.rcParams['font.size'] = 8
fig, axs = plt.subplots(3, 1)
fig.tight_layout(rect=(0, 0, 1, 0.95), pad=3.0)
fig.suptitle("$t^{1 - \\alpha}y'(t)=-y(t-1)$ solved by jitcdde")

alphas = [1, 0.66, 0.33, 0]
colors = ['red', 'blue', 'green', 'pink']
equations = [[-t ** (alpha - 1) * y(0, t - 1.)] for alpha in alphas]

for equation, color, alpha in zip(equations, colors, alphas):
	dde = jitcdde(equation)
	ts = np.linspace(0, 20, 2000)
	for i in range(3):
		dde.constant_past([i - 1])
		dde.integrate_blindly(1)
		ys = []
		for t in ts:
			ys.append(dde.integrate(t))
		axs[i].plot(ts, ys, color=color, linewidth=1, label='$\\alpha=$' + str(alpha))
		axs[i].set_title(f'$ihf(t)={i - 1}$')
		axs[i].legend()
		axs[i].margins(0, 0.1)
plt.show()

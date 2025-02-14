import matplotlib.pyplot as plt
import numpy as np
from jitcdde import jitcdde, t, y


def ihf1(t):
    return [np.exp(-t) - 1]


def ihf2(t):
    return [np.exp(t) - 1]


ihfs = [ihf1, ihf2]

plt.rcParams['font.size'] = 8
fig, axs = plt.subplots(2, 2)
fig.tight_layout(rect=(0, 0, 1, 0.95), pad=3.0)
fig.suptitle("$y'(t)=-y(t-2)$ solved by jitcdde")

alphas = [1, 0.66, 0.33, 0]
colors = ['red', 'blue', 'green', 'pink']
equations = [[-t ** (alpha - 1) * y(0, t - 2.)] for alpha in alphas]

for equation, color, alpha in zip(equations, colors, alphas):
    dde = jitcdde(equation)
    for i, rb in enumerate([4, 60]):
        ts = np.linspace(0, rb, 2000)
        for j, eqText in enumerate(["e^{{-t}} - 1", "e^{{t}} - 1"]):
            dde.past_from_function(ihfs[j])
            dde.integrate_blindly(2)
            ys = []
            for t in ts:
                ys.append(dde.integrate(t))
            axs[i, j].plot(ts, ys, color=color, linewidth=1,
                           label='$\\alpha=$' + str(alpha))
            axs[i, j].set_title(f'$ihf(t)={eqText}, t \\in [0, {rb}]$')
            axs[i, j].legend()
            axs[i, j].margins(0, 0.1)
plt.show()

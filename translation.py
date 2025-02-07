import numpy as np
import matplotlib.pyplot as plt

omega = 0.8
delta = 0.1

A = np.matrix([[0, 1], [-omega ** 2, -2 * delta * omega]])
B = np.c_[[0, 1]]
Sf = np.matrix('3 0; 0 1')
Q = 0.001 * np.identity(2)
R = 1
tf = 20
dt = 0.001
alpha = 0.33
t_dis = np.arange(0, tf + dt, dt)

S = np.empty((t_dis.size, 2, 2))
S[-1] = Sf

for i in range(t_dis.size - 1, 0, -1):
    dSdt = -(dt * (i + 1)) ** (alpha - 1) * (A.T @ S[i] + S[i] @ A - S[i] @ B @ B.T @ S[i] + Q)
    S[i - 1] = S[i] - dt * dSdt;
    # print(S[i])
x = np.empty((t_dis.size, 2, 1))
x[0] = np.c_[[10, 3]]

u = np.empty(t_dis.size - 1)

for i in range(0, t_dis.size - 1):
    u[i] = -(B.T @ S[i] @ x[i]).item()
    x[i + 1] = x[i] + (dt * (i + 2)) ** (alpha - 1) * dt * (A @ x[i] + B * u[i])
fig, axs = plt.subplots(1, 1)
axs.plot(t_dis[:-1], u)
plt.show()

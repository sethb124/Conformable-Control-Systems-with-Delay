# this file is where I started translating the mathematica code

import control as ct

# natural frequency
omega = 0.8

# damping
delta = 0.1

tf = 20
dt = 0.001
alpha = 1

#initial x value
x0 = [[10], [3]]

A = [[0, 1], [-0.64, -0.16]]
B = [[0], [1]]
Sf = [[3, 0], [0, 1]]
Q = [[0.001, 0], [0, 0.001]]
R = 1

print(S)

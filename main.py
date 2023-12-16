import numpy as np
import matplotlib.pyplot as plt

H, m, T = 0.5, 1, 15
k = [5, 50, 0.5, 5]
h = [0.01, 0.001, 0.01, 0.01]
y = [20, 20, 20, 7]
z = [0, 0, 0, 13]

def funcOption5(x):
    return 0

def funcIndependent(x):
    return np.cos(-0.4*x)*np.tan(x)

def runge_kutta(k, h, index, y, z):
    yk = y
    zk = z

    y_values = [y]
    z_values = [z]

    Func = funcIndependent if index == 3 else funcOption5

    for x in np.arange(0, T, h):
        k1 = zk * h
        L1 = h * ((Func(x) - H * zk - k * yk) / m)

        k2 = h * (zk + L1 / 2.0)
        L2 = h * ((Func(x+h/2) - H * (zk + L1 / 2.0) - k * (yk + k1 / 2.0)) / m)

        k3 = h * (zk + L2 / 2.0)
        L3 = h * ((Func(x+h/2) - H * (zk + L2 / 2.0) - k * (yk + k2 / 2.0)) / m)

        k4 = h * (zk + L3)
        L4 = h * ((Func(x+h) - H * (zk + L3) - k * (yk + k2)) / m)

        dy = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        dz = (L1 + 2.0 * L2 + 2.0 * L3 + L4) / 6.0

        tky = yk + dy
        tkz = zk + dz

        y_values.append(tky)
        z_values.append(tkz)

        yk = tky
        zk = tkz

    return y_values, z_values

plt.figure(figsize=(12, 9))

for i in range(len(k)):
    y_values, z_values = runge_kutta(k[i], h[i], i, y[i], z[i])
    plt.subplot(len(k), 1, i+1)
    plt.plot(y_values, marker='o')
    plt.plot(z_values, marker='o')
    plt.title(f'Графік для завдання = {i+1}')
    plt.grid(True)

plt.tight_layout()
plt.show()
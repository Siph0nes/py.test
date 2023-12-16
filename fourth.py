import numpy as np
import matplotlib.pyplot as plt

h, H, k, m, y0, z0, T = 0.01, 0.5, 5.0, 1.0, 7.0, 13.0, 15.0

def funcIndependent(x):
    return np.cos(-0.4*x)*np.tan(x)

def runge_kutta():
    yk = y0
    zk = z0

    x_values = [y0]
    z_values = [z0]

    for x in np.arange(0, T, h):
        k1 = zk * h
        L1 = h * ((funcIndependent(x) - H * zk - k * yk) / m)

        k2 = h * (zk + L1 / 2.0)
        L2 = h * ((funcIndependent(x+h/2) - H * (zk + L1 / 2.0) - k * (yk + k1 / 2.0)) / m)

        k3 = h * (zk + L2 / 2.0)
        L3 = h * ((funcIndependent(x+h/2) - H * (zk + L2 / 2.0) - k * (yk + k2 / 2.0)) / m)

        k4 = h * (zk + L3)
        L4 = h * ((funcIndependent(x+h) - H * (zk + L3) - k * (yk + k2)) / m)

        dy = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        dz = (L1 + 2.0 * L2 + 2.0 * L3 + L4) / 6.0

        tky = yk + dy
        tkz = zk + dz

        x_values.append(tky)
        z_values.append(tkz)

        yk = tky
        zk = tkz

    return x_values, z_values

x_values, z_values = runge_kutta()

plt.figure(figsize=(10, 5))

plt.plot(x_values, marker='o')
plt.plot(z_values, marker='o')
plt.title('Графік для самостійного варіанту')
plt.grid(True)

plt.tight_layout()
plt.show()

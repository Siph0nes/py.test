import numpy as np
import matplotlib.pyplot as plt

h = 0.01
H = 0.5
k = 5.0
m = 1.0
y0 = 7
z0 = 13
T = 15

def funcIndependent(tk):
    return np.cos(-0.4*tk)*np.tan(tk)

def runge_kutta():
    yk = y0
    zk = z0

    x_values = [y0]
    z_values = [z0]

    for tk in np.arange(0, T, h):
        k1 = zk * h
        L1 = h * ((funcIndependent(tk) - H * zk - k * yk) / m)

        k2 = h * (zk + L1 / 2.0)
        L2 = h * ((funcIndependent(tk) - H * (zk + L1 / 2.0) - k * (yk + k1 / 2.0)) / m)

        k3 = h * (zk + L2 / 2.0)
        L3 = h * ((funcIndependent(tk) - H * (zk + L2 / 2.0) - k * (yk + k2 / 2.0)) / m)

        k4 = h * (zk + L3)
        L4 = h * ((funcIndependent(tk) - H * (zk + L3) - k * (yk + k2)) / m)

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

plt.plot(x_values, label='x(t) for k=5', marker='o')
plt.plot(z_values, label='z(t) for k=5', marker='o')
plt.xlabel('Time Steps')
plt.ylabel('Values')
plt.title('Runge-Kutta 4th Order Method for k = 5')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

h, h2, H, m, y0, z0, T = 0.01, 0.001, 0.5, 1, 20, 0, 15
k1, k2, k3 = 5, 50, 0.5

def funcOption5(tk):
    return 0

def runge_kutta(k, h):
    yk = y0
    zk = z0

    x_values = [y0]
    z_values = [z0]

    for tk in np.arange(0, T, h):
        k1 = zk * h
        L1 = h * ((funcOption5(tk) - H * zk - k * yk) / m)

        k2 = h * (zk + L1 / 2.0)
        L2 = h * ((funcOption5(tk) - H * (zk + L1 / 2.0) - k * (yk + k1 / 2.0)) / m)

        k3 = h * (zk + L2 / 2.0)
        L3 = h * ((funcOption5(tk) - H * (zk + L2 / 2.0) - k * (yk + k2 / 2.0)) / m)

        k4 = h * (zk + L3)
        L4 = h * ((funcOption5(tk) - H * (zk + L3) - k * (yk + k2)) / m)

        dy = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        dz = (L1 + 2.0 * L2 + 2.0 * L3 + L4) / 6.0

        tky = yk + dy
        tkz = zk + dz

        x_values.append(tky)
        z_values.append(tkz)

        yk = tky
        zk = tkz

    return x_values, z_values

x_values1, z_values1 = runge_kutta(k1, h)

plt.figure(figsize=(12, 15))
plt.subplot(3, 1, 1)
plt.plot(x_values1, label=f'x(t) for k={k1}', marker='o')
plt.plot(z_values1, label=f'z(t) for k={k1}', marker='o')
plt.xlabel('Time Steps')
plt.ylabel('Values')
plt.title(f'Runge-Kutta 4th Order Method for k = {k2}')
plt.legend()
plt.grid(True)

x_values2, z_values2 = runge_kutta(k2, h2)

plt.subplot(3, 1, 2)
plt.plot(x_values2, label=f'x(t) for k={k2}', marker='o')
plt.plot(z_values2, label=f'z(t) for k={k2}', marker='o')
plt.xlabel('Time Steps')
plt.ylabel('Values')
plt.title(f'Runge-Kutta 4th Order Method for k = {k2}')
plt.legend()
plt.grid(True)

x_values3, z_values3 = runge_kutta(k3, h)
plt.subplot(3, 1, 3)
plt.plot(x_values3, label=f'x(t) for k={k3}', marker='o')
plt.plot(z_values3, label=f'z(t) for k={k3}', marker='o')
plt.xlabel('Time Steps')
plt.ylabel('Values')
plt.title(f'Runge-Kutta 4th Order Method for k = {k3}')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
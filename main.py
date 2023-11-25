import numpy as np
import matplotlib.pyplot as plt

# Constants
h = 0.1
H = 0.5
k1 = 5
k2 = 50
k3 = 0.5
m = 1
x0 = 20
z0 = 0
N = 20
def f(tk, x, z):
    return z

def g(t, x, z, k):
    return -k/m * x

def runge_kutta_4(x0, z0, h, N, k):
    tk = 0
    x = x0
    z = z0
    x_values = [x0]
    z_values = [z0]

    for i in range(N):
        k1x = h * f(tk, x, z)
        k1z = h * g(tk, x, z, k)

        k2x = h * f(tk + h/2, x + k1x/2, z + k1z/2)
        k2z = h * g(tk + h/2, x + k1x/2, z + k1z/2, k)

        k3x = h * f(tk + h/2, x + k2x/2, z + k2z/2)
        k3z = h * g(tk + h/2, x + k2x/2, z + k2z/2, k)

        k4x = h * f(tk + h, x + k3x, z + k3z)
        k4z = h * g(tk + h, x + k3x, z + k3z, k)

        x += (k1x + 2 * k2x + 2 * k3x + k4x) / 6
        z += (k1z + 2 * k2z + 2 * k3z + k4z) / 6

        x_values.append(x)
        z_values.append(z)

    return x_values, z_values

k = 5
x_values1, z_values1 = runge_kutta_4(x0, z0, h, N, k)

plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(x_values1, label='x(t) for k=5')
plt.plot(z_values1, label='z(t) for k=5')
plt.xlabel('Time Steps')
plt.ylabel('Values')
plt.title('Runge-Kutta 4th Order Method for k = 5')
plt.legend()
plt.grid(True)

k = 50
x_values2, z_values2 = runge_kutta_4(x0, z0, h, N, k)

plt.subplot(3, 1, 2)
plt.plot(x_values2, label='x(t) for k=50')
plt.plot(z_values2, label='z(t) for k=50')
plt.xlabel('Time Steps')
plt.ylabel('Values')
plt.title('Runge-Kutta 4th Order Method for k = 50')
plt.legend()
plt.grid(True)

k = 0.5
x_values3, z_values3 = runge_kutta_4(x0, z0, h, N, k)

plt.subplot(3, 1, 3)
plt.plot(x_values3, label='x(t) for k=0.5')
plt.plot(z_values3, label='z(t) for k=0.5')
plt.xlabel('Time Steps')
plt.ylabel('Values')
plt.title('Runge-Kutta 4th Order Method for k = 0.5')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
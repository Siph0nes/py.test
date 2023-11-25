import numpy as np
import matplotlib.pyplot as plt

h = 0.1
H = 0.5
k1 = 5
m = 1
x0 = 7
z0 = 13
N = 20
def f(tk, x, z):
    return np.cos(-0.4*tk)*np.exp(tk)*np.tan(tk)

def g(t, x, z, k):
    return -k/m * x

def runge_kutta_4(x0, z0, h, N, k):

    x = x0
    z = z0
    x_values = [x0]
    z_values = [z0]

    for tk, i in zip(np.arange(0.1, 2, 0.1), range(N)):
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

x_values1, z_values1 = runge_kutta_4(x0, z0, h, N, k1)

plt.plot(x_values1, label='x(t)')
plt.plot(z_values1, label='z(t)')
plt.xlabel('Time Steps')
plt.ylabel('Values')
plt.title('Runge-Kutta 4th Order Method for k = 5')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
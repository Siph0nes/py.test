import numpy as np
import matplotlib.pyplot as plt

h = 0.1
H = 0.5
k1 = 5
k2 = 50
k3 = 0.5
m = 1
x0 = 20
z0 = 0
N = 20
def f(tk):
    return 0

def K(zk):
    return zk * h
def L(tk,zk,xk, k):
    return h*((f(tk)-H*zk-xk*k)/m)
def dx(K1, K2, K3, K4):
    return (K1 + 2 * K2 + 2 * K3 + K4)/6

def dz(L1, L2, L3, L4):
    return (L1 + 2 * L2 + 2 * L3 + L4) / 6

def runge_kutta(h, H, k, m, x0,z0):
    xk = x0
    zk = z0

    q_values = []

    x_values = [x0]
    z_values = [z0]

    for tk in np.arange(0, 2, 0.1):
        K1 = K(zk)
        L1 = L(tk, zk, xk, k)

        xk2 = xk + h/2
        zk2 = zk + L1/2

        K2 = K(zk2)
        L2 = L(tk, zk2, xk2, k)

        xk3 = xk2 + h / 2
        zk3 = zk2 + L2 / 2

        K3 = K(zk3)
        L3 = L(tk, zk3, xk3, k)

        xk4 = xk3 + h / 2
        zk4 = zk3 + L3 / 2

        K4 = K(zk4)
        L4 = L(tk, zk4, xk4, k)

        dx1 = dx(K1,K2,K3,K4)
        dz1 = dz(L1, L2, L3, L4)

        x1k = dx1 + xk
        z1k = dz1 + zk

        x_values.append(x1k)
        z_values.append(z1k)

        xk = x1k
        zk = z1k


    return x_values, z_values

x_values1, z_values1 = runge_kutta(h, H, k1, m, x0,z0)

plt.figure(figsize=(12, 15))
plt.subplot(3, 1, 1)
plt.plot(x_values1, label='x(t) for k=5', marker='o')
plt.plot(z_values1, label='z(t) for k=5', marker='o')
plt.xlabel('Time Steps')
plt.ylabel('Values')
plt.title('Runge-Kutta 4th Order Method for k = 5')
plt.legend()
plt.grid(True)

x_values2, z_values2 = runge_kutta(h, H, k2, m, x0,z0)

plt.subplot(3, 1, 2)
plt.plot(x_values2, label='x(t) for k=50', marker='o')
plt.plot(z_values2, label='z(t) for k=50', marker='o')
plt.xlabel('Time Steps')
plt.ylabel('Values')
plt.title('Runge-Kutta 4th Order Method for k = 50')
plt.legend()
plt.grid(True)

x_values3, z_values3 = runge_kutta(h, H, k3, m, x0,z0)

plt.subplot(3, 1, 3)
plt.plot(x_values3, label='x(t) for k=0.5', marker='o')
plt.plot(z_values3, label='z(t) for k=0.5', marker='o')
plt.xlabel('Time Steps')
plt.ylabel('Values')
plt.title('Runge-Kutta 4th Order Method for k = 0.5')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
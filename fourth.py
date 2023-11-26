import numpy as np
import matplotlib.pyplot as plt

h = 0.1
H = 0.5
k = 5
m = 1
x0 = 7
z0 = 13

def f(tk):
    return np.cos(-0.4*tk)*np.exp(tk)*np.tan(tk)

def K(zk):
    return zk * h
def L(tk,zk,xk):
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

    for tk in np.arange(0.1, 2, 0.1):
        K1 = K(zk)
        L1 = L(tk, zk, xk)

        xk2 = xk + h/2
        zk2 = zk + L1/2

        K2 = K(zk2)
        L2 = L(tk, zk2, xk2)

        xk3 = xk2 + h / 2
        zk3 = zk2 + L2 / 2

        K3 = K(zk3)
        L3 = L(tk, zk3, xk3)

        xk4 = xk3 + h / 2
        zk4 = zk3 + L3 / 2

        K4 = K(zk4)
        L4 = L(tk, zk4, xk4)

        dx1 = dx(K1,K2,K3,K4)
        dz1 = dz(L1, L2, L3, L4)

        x1k = dx1 + xk
        z1k = dz1 + zk

        q = abs((K2 - K3)/(K1 - K2))
        q_values.append(q)
        q = 0

        x_values.append(x1k)
        z_values.append(z1k)

        xk = x1k
        zk = z1k


    return x_values, z_values

x, z = runge_kutta(h, H, k, m, x0,z0)


plt.plot(x, label='x(t) for k=5', marker='o')
plt.plot(z, label='z(t) for k=5', marker='o')
plt.xlabel('Time Steps')
plt.ylabel('Values')
plt.title('Runge-Kutta 4th Order Method for k = 5')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

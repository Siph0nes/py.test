import numpy as np
import matplotlib.pyplot as plt

H = 1
k = 1
m = 2
q = 0.04
h = 0.1
T = 20

x0 = 0
v0 = 0

def f(t):
    return -np.cos(0.5*t)

def dxdt(t, x, v):
    return v

def dvdt(t, x, v):
    return (f(t) - 2*q*v - (k/m)*x)

def runge_kutta(t, h, x, v):
    for i in range(len(t)-1):
        k1x = h * dxdt(t[i], x[i], v[i])
        k1v = h * dvdt(t[i], x[i], v[i])

        k2x = h * dxdt(t[i] + h/2, x[i] + k1x/2, v[i] + k1v/2)
        k2v = h * dvdt(t[i] + h/2, x[i] + k1x/2, v[i] + k1v/2)

        k3x = h * dxdt(t[i] + h/2, x[i] + k2x/2, v[i] + k2v/2)
        k3v = h * dvdt(t[i] + h/2, x[i] + k2x/2, v[i] + k2v/2)

        k4x = h * dxdt(t[i] + h, x[i] + k3x, v[i] + k3v)
        k4v = h * dvdt(t[i] + h, x[i] + k3x, v[i] + k3v)

        x[i+1] = x[i] + (k1x + 2*k2x + 2*k3x + k4x) / 6
        v[i+1] = v[i] + (k1v + 2*k2v + 2*k3v + k4v) / 6

    return x, v

t = np.arange(0, T, h)

x = np.zeros(len(t))
v = np.zeros(len(t))

x[0] = x0
v[0] = v0

x, v = runge_kutta(t, h, x, v)

plt.figure(figsize=(10,5))
plt.plot(t, x, label='x(t)')
plt.plot(t, v, label='v(t)')
plt.legend(loc='best')
plt.xlabel('Час')
plt.ylabel('Значення')
plt.title('Рішення задачі Коші методом Рунге-Кутта')
plt.grid(True)

plt.xlim(0, 1)
plt.ylim(0, -1)

plt.show()

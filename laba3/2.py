import scipy.linalg as sp
import numpy as np
import matplotlib.pyplot as plt

n = 20
delta = 0.05

A = sp.hilbert(n)
x_true = np.ones(n)
b = A @ x_true

sigma = np.random.normal(0, 1, n)
b_delta = b + delta * sigma

x_noisy = np.linalg.solve(A, b_delta)

plt.plot(x_true)
plt.plot(x_noisy)
plt.show()
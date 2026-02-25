import scipy.linalg as sp
import numpy as np
import matplotlib.pyplot as plt

n = 20
delta = 0.05
A = sp.hilbert(n)
U, S, V = np.linalg.svd(A)

x_true = np.ones(n)
b = A @ x_true
sigma = np.random.normal(0, 1, n)
b_delta = b + delta * sigma

# plt.plot(S, 'k-s')
# plt.show()

eigvals = np.linalg.eigvals(A)
lambda_min = np.min(eigvals)
lambda_max = np.max(eigvals)

u = np.zeros(n)
tau = 1 / (lambda_min + lambda_max)
max_iter = 20000
eps = 1e-6

for k in range(max_iter):
    rk = A @ u - b_delta
    u_new = u - tau * rk

    if np.linalg.norm(A @ u_new - b_delta) < delta * np.sqrt(n):
        print('stop', k)
        break

    u = u_new

x_iter = u

print(u)
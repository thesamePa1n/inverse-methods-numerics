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

alphas = np.logspace(-8, -2, 50)

J_values = []
residual_values = []
reg_values = []

E = np.eye(n)

for alpha in alphas:
    M = A.T @ A + alpha * E
    rhs = A.T @ b_delta
    x_alpha = np.linalg.solve(M, rhs)

    residual = A @ x_alpha - b_delta
    residual_norm2 = np.linalg.norm(residual)**2
    reg_norm2 = alpha * np.linalg.norm(x_alpha)**2

    J = residual_norm2 + reg_norm2

    residual_values.append(residual_norm2)
    reg_values.append(reg_norm2)
    J_values.append(J)

M = A.T @ A + 1e-3*E
rhs = A.T @ b_delta
x_alpha = np.linalg.solve(M, rhs)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(alphas, J_values, 'o-')
plt.xscale('log') 
plt.yscale('log')

plt.subplot(1, 3, 2)
plt.plot(alphas, residual_values, 'o-')
plt.xscale('log')
plt.yscale('log')

plt.subplot(1, 3, 3)
plt.plot(alphas, reg_values, 'o-')
plt.xscale('log')
plt.yscale('log')

plt.tight_layout()

plt.figure()

plt.plot(x_alpha)

plt.tight_layout()
plt.show()
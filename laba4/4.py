import scipy.linalg as sp
import numpy as np
import matplotlib.pyplot as plt

n = 20
delta = 0.05
A = sp.hilbert(n)
AA = A.T @ A
U, S, V = np.linalg.svd(A)

x_true = np.ones(n)
b = A @ x_true
sigma = np.random.normal(0, 1, n)
b_delta = b + delta * sigma
bb_delta = A.T @ b_delta

u = np.zeros(n)
max_iter = 20000

for k in range(max_iter):
    r = AA @ u - bb_delta
    Ar = AA @ r

    tau = (Ar @ r) / (Ar @ Ar)

    u_new = u - tau * r

    if np.linalg.norm(AA @ u_new - bb_delta) < delta * np.sqrt(n):
        print("stop", k)
        break

    u = u_new

print(u)
import numpy as np
import scipy.linalg as sp
import matplotlib.pyplot as plt

sizes = [10, 20, 100]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, n in enumerate(sizes):
    H = sp.hilbert(n)
    eigvals = np.linalg.eigvals(H)

    axes[i].plot(eigvals, 'o-')
    axes[i].set_title(f'Собственные значения, n = {n}')
    axes[i].set_yscale('log') 
    axes[i].grid(True)

plt.tight_layout()
plt.show()

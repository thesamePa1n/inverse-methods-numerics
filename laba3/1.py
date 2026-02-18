import scipy.linalg as sp
import numpy as np
import matplotlib.pyplot as plt

sizes = [10, 20, 100]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, n in enumerate(sizes):
    hilbert_matrix = sp.hilbert(n)
    U, S, V = np.linalg.svd(hilbert_matrix)

    axes[i].plot(S, 'k-s')
    axes[i].set_title(f'n = {n}')

plt.show()
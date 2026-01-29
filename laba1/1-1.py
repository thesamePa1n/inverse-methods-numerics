import numpy as np
import matplotlib.pyplot as plt

# Строим равномерную сетку на отрезке [0,1]
N = 100                     # кол-во интервалов
x = np.linspace(0, 1, N+1)  # узлы сетки
h = 1 / N                   # шаг сетки

# Собираем матрицу для внутренних узлов, их кол-во N-1
A = np.zeros((N-1, N-1))

for i in range(1, N-2):
    A[i, i-1] = -1
    A[i, i] = 2
    A[i, i+1] = -1

A[0, 0] = 2
A[0, 1] = -1
A[-1, -2] = -1
A[-1, -1] = 2

A /= h**2

# Правая часть
f = 1.0
b = np.ones(N-1) * f

# Вектор решения, включая граничные значения
u = np.zeros_like(x)

# Граничные условия
u0 = 0
u1 = 1
u[0] = u0
u[-1] = u1

# Корректировка правой части
b[0] += u0 / h**2
b[-1] += u1 / h**2

# Решение системы Au = b
def solver(A, b):
    return np.linalg.solve(A, b)

u_inner = solver(A, b)

# Собираем решение
u[1:-1] = u_inner


# Строим график
fig, ax = plt.subplots()
plt.plot(x, u)

plt.show()

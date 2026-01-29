import numpy as np
import matplotlib.pyplot as plt


N = 100                     
x = np.linspace(0,1,N+1)    
h = 1/N                     

A = np.zeros((N,N))
for i in range(1,N-1):
    A[i,i-1] = -1   
    A[i,i] = 2  
    A[i,i+1] = -1

A[0,0] = 2
A[0,1] = -1
A[-1,-2] = -1
A[-1,-1] = 1

A /= h**2 
f = 1.0
b = np.ones(N) * f

u = np.zeros_like(x)
u0 = 0

u[0] = u0

b[0] += u[0]/h**2
b[-1] = 0.5*f

def solver(A, b):    
  return np.linalg.solve(A, b)

u_ = solver(A, b)
u[1:] = u_

fig, ax =plt.subplots()
plt.plot(x, u)
plt.show()
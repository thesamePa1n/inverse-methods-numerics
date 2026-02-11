import numpy as np
import matplotlib.pyplot as plt

L = 1.0
T = 1.0
q = 0.1
f = 1.0

Nx = 49         
Nt = 500         
h = L / Nx
tau = T / Nt     

r = tau / (h**2)

x = np.linspace(0, L, Nx)

u = np.zeros((Nt + 1, Nx))  
u[0, :] = 0.0             

def progonka(a, b, c, d):
    n = len(d)
    
    alpha = np.zeros(n)
    beta = np.zeros(n)
    
    alpha[0] = -c[0] / b[0]
    beta[0] = d[0] / b[0]
    
    for i in range(1, n):
        denominator = b[i] + a[i] * alpha[i-1]
        alpha[i] = -c[i] / denominator
        beta[i] = (d[i] - a[i] * beta[i-1]) / denominator
    
    x = np.zeros(n)
    x[-1] = beta[-1]
    
    for i in range(n-2, -1, -1):
        x[i] = alpha[i] * x[i+1] + beta[i]
    
    return x

x_ = 0.5
idx = Nx // 2
u_ = np.zeros(Nt + 1)
u_[0] = u[0, idx]

for n in range(Nt):
    a = np.zeros(Nx)  
    b = np.zeros(Nx) 
    c = np.zeros(Nx) 
    d = np.zeros(Nx)  
    
    a[0] = 0.0
    b[0] = 1.0
    c[0] = 0.0
    d[0] = 0.0
    
    for i in range(1, Nx-1):
        a[i] = -r
        b[i] = 1.0 + 2.0*r + q*tau
        c[i] = -r
        d[i] = u[n, i] + tau * f  
    
    a[Nx-1] = 0.0
    b[Nx-1] = 1.0
    c[Nx-1] = 0.0
    d[Nx-1] = 1.0
    
    u_new = progonka(a, b, c, d)
    u[n+1, :] = u_new
    u_[n+1] = u_new[idx]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(x, u[-1, :])
axes[1].plot(np.linspace(0, T, Nt+1), u_)

plt.show()

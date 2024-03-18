import numpy as np
import matplotlib.pyplot as plt
import time

start = time.time()
domain = int(1e6)
samples = 100
t = np.linspace(-np.pi, np.pi, domain)

a0 = (2*np.pi**2) / 3
signal = a0 + np.zeros(domain)
for n in range(1, samples + 1):
    an = (4*(-1)**(n+1)) / n**2
    signal += an*np.cos(n*t)
end = time.time()

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
ax.plot(t, signal, label='Fourier Series')  
ax.plot(t, -t**2 + np.pi**2, '--', color="orange", label=r'Real Function: $-t² + \pi^2$')

ax.set_xlabel('t[s]', fontsize=15)
ax.set_ylabel('Amplitude', fontsize=15)

ax.legend(loc = 'upper right')
plt.grid(True)
plt.show()

print(end - start)
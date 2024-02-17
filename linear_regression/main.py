import function
import numpy as np
import matplotlib.pyplot as plt

parameters = np.array([3, 2])
x = np.linspace(-2, 2, 101)
y = function.f(x, parameters)

y_noise = function.noise(y)

mat, a0, a1 = function.solution_space(x, y_noise)

min_index = np.unravel_index(np.argmin(mat, axis=None), mat.shape)
a0_min, a1_min = a0[min_index], a1[min_index]

plt.imshow(mat, extent= [-5, 5, 5, -5], aspect= "auto")
plt.scatter(a0_min, a1_min, color = 'k')
plt.show()

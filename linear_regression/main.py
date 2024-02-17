import function
import numpy as np
import matplotlib.pyplot as plt

parameters = np.array([-2, 1])
x = np.linspace(-2, 2, 101)
y = function.f(x, parameters)

y_noise = function.noise(y)

mat = function.solution_space(x, y)

# plt.imshow(mat, extent= [-5, 5, 5, -5])
# plt.show()

a0_ind, a1_ind = np.where(mat == np.min(mat))
print(a0_ind, a1_ind)



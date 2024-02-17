import numpy as np 
import matplotlib.pyplot as plt 

#   defining a polynomial
def f(x, parameters):
    
    function = 0.0
    for n, p in enumerate(parameters):
        function += p*x**n              
    
    return function

#   applying gaussian noise
def noise(y):
    std = 0.5*np.abs(y)
    noise = std*np.random.rand(len(y))
    y_noise = y + noise
    return y_noise

#   plotting graph
def plot_graph(x, y):
    fig, ax = plt.subplots(1, 1, figsize = (10,5))
    
    ax.plot(x,y)

    fig.tight_layout
    plt.grid(True)
    plt.show()

# euclidian norm
# creating solution space of a0, a1
def solution_space(x, y):
    
    n = 1001

    a0 = np.linspace(-5, 5, n)
    a1 = np.linspace(-5, 5, n)

    a0, a1 = np.meshgrid(a0, a1)

    mat = np.zeros([n,n])

    for i in range(n):
        for j in range(n):
            y_p = a0[i,j] + a1[i,j]*x

            mat[i,j] = np.sqrt(np.sum((y - y_p)**2))

    return mat

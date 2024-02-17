import numpy as np 
import matplotlib.pyplot as plt 

class linear_regression():

    def __init__(self):

        self.parameters = np.array([3, 2])
        self.x = np.linspace(-2, 2, 101)
        self.y = f(self.x, self.parameters)         

    def noise(self):
#   applying gaussian noise

        self.std = 0.5*np.abs(self.y)
        self.noise = self.std*np.random.rand(len(self.y))
        self.y_noise = self.y + self.noise

    def plot_graph(self):
#   plotting graph
        
        fig, ax = plt.subplots(1, 1, figsize = (10,5))
    
        ax.plot(self.x, self.y)

        fig.tight_layout
        plt.grid(True)
        plt.show()

    def solution_space(self):
#   euclidian norm
#   creating solution space of a0, a1
    
        self.n = 1001

        self.a0 = np.linspace(-5, 5, self.n)
        self.a1 = np.linspace(-5, 5, self.n)

        self.a0, self.a1 = np.meshgrid(self.a0, self.a1)

        self.mat = np.zeros([self.n, self.n])

        for i in range(self.n):
            for j in range(self.n):
                self.y_p = self.a0[i,j] + self.a1[i,j]*self.x

                self.mat[i,j] = np.sqrt(np.sum((self.y - self.y_p)**2))

        self.min_index = np.unravel_index(np.argmin(self.mat, axis=None), self.mat.shape)
        
        self.a0_min, self.a1_min = self.a0[self.min_index], self.a1[self.min_index]

    def plot_mesh(self):
            
            plt.imshow(self.mat, extent= [-5, 5, 5, -5], aspect= "auto")
            plt.scatter(self.a0_min, self.a1_min, color = 'k')
            plt.show()

def f(x, parameters):
#   defining a polynomial
        
        function = 0.0
        for n, p in enumerate(parameters):
            function += p*x**n 

        return function

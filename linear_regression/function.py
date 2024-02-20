import numpy as np 
import matplotlib.pyplot as plt 

class LinearRegressionBase():
    def __init__(self, parameters= np.array([3, 2]), x= np.linspace(-2, 2, 101)):
        self.parameters = parameters
        self.x = x
        self.y = self.f(self.x, self.parameters)
        self.y_noise = self.noise(self.y)

    def f(self, x, parameters):
        function = 0.0
        for n, p in enumerate(parameters):
            function += p*x**n
        return function

    def noise(self, y):
        std = 0.5*np.abs(y)
        noises = std*np.random.rand(len(y))
        return y + noises

    def plot_graph(self, y):

        fig, ax = plt.subplots(1, 1, figsize = (10,5))

        ax.plot(self.x, y, color= "k")
        ax.scatter(self.x, self.y_noise)

        fig.tight_layout()
        plt.grid(True)
        plt.show()

class LinearRegressionBruteForce(LinearRegressionBase):
    def solution_space(self):
        n = 1001
        self.a0 = np.linspace(-5, 5, n)
        self.a1 = np.linspace(-5, 5, n)
        self.a0, self.a1 = np.meshgrid(self.a0, self.a1)

        self.mat = np.zeros([n, n])
        for i in range(n):
            for j in range(n):
                y_p = self.a0[i,j] + self.a1[i,j]*self.x

                self.mat[i,j] = np.sqrt(np.sum((self.y_noise - y_p)**2))

        self.min_index = np.unravel_index(np.argmin(self.mat, axis=None), self.mat.shape)

        return self.a0[self.min_index], self.a1[self.min_index]

    def plot_mesh(self):

        plt.imshow(self.mat, extent= [-5, 5, 5, -5], aspect= "auto")
        plt.scatter(self.a0[self.min_index], self.a1[self.min_index], color = 'k')

        plt.show()

class LinearRegressionMatrix(LinearRegressionBase):
    def linear_regression_solver(self):

        d = self.y_noise
        G = np.zeros((len(d), len(self.parameters)))

        for n in range(len(self.parameters)): 
            G[:,n] = self.x**n

        GTG = np.dot(G.T, G)
        GTD = np.dot(G.T, d)

        self.m = np.linalg.solve(GTG, GTD)

    def y_regression(self):
        self.y_reg = self.f(self.x, self.m)

        print(f"Regression Parameters: {self.m}")
import numpy as np 
import matplotlib.pyplot as plt 

class LinearRegressionBase():
    def __init__(self, parameters= np.array([1/3, 2/3]), x= np.linspace(-2, 2, 101)):
        self.parameters = parameters
        self.x = x
        self.y = self.f(self.x, self.parameters)
        self.y_noise = self.noise(self.y)

    def f(self, x, parameters):
        function = 0.0
        for n, p in enumerate(parameters):
            function += p*x**n
        return function

    def noise(self, y, k):
        std = k*np.abs(y)
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

                self.mat[i,j] = np.sqrt(np.sum((self.y_noise - y_p)**2)) # euclian norm

        self.min_index = np.unravel_index(np.argmin(self.mat, axis=None), self.mat.shape)

        print(f"{self.a0[self.min_index], self.a1[self.min_index]} euclian norm")

    def plot_mesh(self):

        plt.figure(figsize= (6,6))

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

class cmp_gather(LinearRegressionBase):
    def __init__(self):
        self.z_true = 700.0
        self.v_true = 4000.0
        self.offset = np.arange(320)*25.0

        self.g_p = self.function(self.offset, self.z_true, self.v_true)
        self.g_p_noise = self.noise(self.g_p, k= 0.05)

        self.m = self.least_square_solver(self.offset**2, self.g_p_noise**2)
        self.t0 = np.sqrt(self.m[0])
        self.v_noise = 1.0 / np.sqrt(self.m[1])
        self.depth = 0.5*self.v_noise*self.t0

    def function(self, offset, z_true, v_true):
        g_p = np.sqrt((offset**2 + 4*z_true**2) / v_true**2)
        return g_p
    
    def least_square_solver(self, offset, d):
        one_matrix = np.ones(len(offset))
        G = np.c_[one_matrix, offset]

        GTG = np.dot(G.T, G)
        GTD = np.dot(G.T, d)

        return np.linalg.solve(GTG, GTD)
    
    def plot_graph(self):
        fig, ax = plt.subplots(ncols= 1, nrows= 1, figsize= (10,5))

        ax.plot(self.offset, self.g_p, label= f"z_true = {self.z_true} and v_true = {self.v_true}")
        ax.plot(self.offset, self.g_p_noise, label= f"z_estimated = {self.depth:.3f} and v_estimated = {self.v_noise:.3f}")

        ax.set_title("CMP Gather", fontsize= 18)
        ax.set_xlabel("Offset [m]", fontsize= 18)
        ax.set_ylabel("TWT [s]", fontsize= 18)

        ax.legend()
        ax.invert_yaxis()

        fig.tight_layout()
        plt.grid(True)
        plt.show()

    def solution_space(self, n= 101):
        self.z = np.linspace(350, 1050, n) #z+-z/2 ; v+-v/2
        self.v = np.linspace(2000, 6000, n) #to centralize values of v and z

        self.v, self.z = np.meshgrid(self.v, self.z)

        self.space_matrix = self.calculate_space_matrix(n)

    def calculate_space_matrix(self, n):
        self.space_matrix = np.zeros([n, n])
        for i in range(n):
            for j in range(n):
                g_p = np.sqrt((self.offset**2 + 4*self.z[i, j]**2) / self.v[i, j]**2)

                self.space_matrix[i,j] = np.sqrt(np.sum((self.g_p_noise - g_p)**2)) # euclian norm

        self.min_index = np.unravel_index(np.argmin(self.space_matrix), self.space_matrix.shape)

        return self.space_matrix

    def plot_mesh(self):
        fig, ax = plt.subplots(ncols= 1, nrows= 1, figsize= (7,7))

        ax.imshow(self.space_matrix, extent= [2000, 6000, 350, 1050], aspect= "auto")

        ax.scatter(self.v_true, self.z_true, color= 'blue', label= "Exact Parameters")
        ax.scatter(self.v[self.min_index], self.z[self.min_index], color = 'k', label= "Estimated Parameters")

        ax.set_title("Solution Space", fontsize= 18)
        ax.set_xlabel("Velocity Space [m/s]", fontsize= 18)
        ax.set_ylabel("Depth Space [m]", fontsize= 18)

        ax.legend()
        fig.tight_layout()
        plt.show()

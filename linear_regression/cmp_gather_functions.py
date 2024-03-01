import numpy as np
import matplotlib.pyplot as plt

class cmp_gather():
    def __init__(self):
        self.z_true = 700.0
        self.v_true = 4000.0
        self.offset = np.arange(320)*25.0

        self.g_p = self.function(self.offset, self.z_true, self.v_true)
        self.g_p_noise = self.noise(self.g_p)

        self.m = self.least_square_solver(self.offset**2, self.g_p_noise**2)
        self.t0 = np.sqrt(self.m[0])
        self.v_noise = 1.0 / np.sqrt(self.m[1])
        self.depth = 0.5*self.v_noise*self.t0

    def function(self, offset, z_true, v_true):
        g_p = np.sqrt((offset**2 + 4*z_true**2) / v_true**2)
        return g_p
    
    def noise(self, y):
        std = 0.05*np.abs(y)
        noises = std*np.random.rand(len(y))
        return y + noises
    
    def least_square_solver(self, offset, d):
        one_matrix = np.ones(len(offset))
        G = np.c_[one_matrix, offset]

        GTG = np.dot(G.T, G)
        GTD = np.dot(G.T, d)

        return np.linalg.solve(GTG, GTD)
    
    def plot_graph(self):
        fig, ax = plt.subplots(ncols= 1, nrows= 1, figsize= (10,5))

        ax.plot(self.offset, self.g_p, label= f"self.z = {self.z_true} and self.v = {self.v_true}")
        ax.plot(self.offset, self.g_p_noise, label= f"self.z = {self.depth:.3f} and self.v = {self.v_noise:.3f}")

        ax.set_title("CMP Gather", fontsize= 18)
        ax.set_xlabel("Offset [m]", fontsize= 18)
        ax.set_ylabel("TWT [s]", fontsize= 18)

        ax.legend()
        ax.invert_yaxis()

        fig.tight_layout()
        plt.grid(True)
        plt.show()

    def solution_space(self, n= 101):
        self.z = np.linspace(350, 1050, n)
        self.v = np.linspace(2000, 6000, n)

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


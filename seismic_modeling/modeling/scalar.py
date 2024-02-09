import numpy as np
import matplotlib.pyplot as plt

class Wavefield_1D():

    wave_type = "1D wave propagation in constant density acoustic isotropic media"
    
    def __init__(self):

        path = r'C:\Users\malum\Desktop\DAVI\GISIS_training\seismic_modeling\parameters.txt'
        parameters = np.loadtxt(path, comments="#")

        self.nt = int(parameters[0]) # 1001.0
        self.dt = parameters[1] # 1e-3
        self.fmax = int(parameters[2]) # 30.0

        self.nz = int(parameters[3]) # 1001.0
        self.dz = int(parameters[4]) # 5.0

        self.depth = np.arange(self.nz)*self.dz
        self.times = np.arange(self.nt)*self.dt

        self.interfaces = np.array([1000, 3000, 5000, 6000])
        self.velocities = np.array([1500, 2500, 3500, 5000, 3500])

        self.model = np.zeros(self.nz)

        self.z_src = np.array([100, 200, 300])
        self.z_rec = np.array([2500, 3500, 4500])

    def get_type(self):
        print(self.wave_type)

    def set_wavelet(self):
    
        t0 = 2.0*np.pi/self.fmax
        fc = self.fmax/(3.0*np.sqrt(np.pi))
        td = np.arange(self.nt)*self.dt - t0

        arg = np.pi*(np.pi*fc*td)**2.0

        self.wavelet = (1.0 - 2.0*arg)*np.exp(-arg)

    def set_model(self):
        for layerId, interface in enumerate(self.interfaces):
            self.model[int(interface/self.dz):] = self.velocities[layerId+1]

    def plot_model(self):
       
        fig, ax = plt.subplots(num = "Velocity Model", figsize = (4, 6), clear = True)

        ax.plot(self.model, self.depth)
        ax.plot(self.model[self.z_src // self.dz], self.z_src, "h", label = "Source", color = 'violet')
        ax.plot(self.model[self.z_rec // self.dz], self.z_rec, "X", label = "Receiver", color = "coral")
        ax.set_title("Velocity Model", fontsize = 18)
        ax.set_xlabel("Velocity [m/s]", fontsize = 15)
        ax.set_ylabel("Depth [m]", fontsize = 15) 
        ax.invert_yaxis()

        ax.legend(loc='upper right')
        
        fig.tight_layout()
        plt.grid(True)
        plt.show()

    def plot_wavefield(self):

        fig, ax = plt.subplots(num = "Wavefield plot", figsize = (8, 8), clear = True)

        ax.imshow(self.P, aspect = "auto", cmap = "Greys")

        ax.set_title("Wavefield plot", fontsize = 18)
        ax.set_xlabel("Time [s]", fontsize = 15)
        ax.set_ylabel("Depth [m]", fontsize = 15)

        fig.tight_layout()
        plt.show()

    def wave_propagation(self):
        
        self.P = np.zeros((self.nz, self.nt))

        sId = int(self.z_src[0] / self.dz)

        for n in range(1, self.nt-1):

            self.P[sId,n] += self.wavelet[n]

            laplacian = get_laplacian_1D(self.P, self.dz, self.nz, n)

            self.P[:, n+1] = (self.dt*self.model)**2 * laplacian + 2.0*self.P[:, n] - self.P[:, n-1]

def get_laplacian_1D(P, dz, nz, time_id):

        d2P_dz2 = np.zeros(nz)

        for i in range(1,nz-1):
            d2P_dz2[i] = (P[i-1, time_id] - 2.0*P[i, time_id] + P[i+1, time_id]) / dz**2.0
        return d2P_dz2

class Wavefield_2D(Wavefield_1D):
    
    def __init__(self):
        super().__init__()
        
        self._type = "2D wave propagation in constant density acoustic isotropic media"    


class Wavefield_3D(Wavefield_2D):
    
    def __init__(self):
        super().__init__()
        
        self._type = "3D wave propagation in constant density acoustic isotropic media"    
from modeling import scalar

def simulation():

    id = 0

    myWave = [scalar.Wavefield_1D(), 
              scalar.Wavefield_2D(),
              scalar.Wavefield_3D()] 

    myWave[id].get_type()

    myWave[id].set_wavelet()
    myWave[id].set_model()

    myWave[id].wave_propagation()

    myWave[id].plot_model()
    myWave[id].plot_wavefield()


if __name__ == "__main__":
    simulation()



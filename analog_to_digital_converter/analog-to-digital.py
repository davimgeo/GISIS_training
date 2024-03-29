import numpy as np
import matplotlib.pyplot as plt

def analog_function(sample_interval, frequency_interval: tuple):
    signal = 0.0
    for k in range(*frequency_interval):
        signal += np.sin(k*np.pi*sample_interval)       # sin(kpix) + sin(nkpix) + sin(2nkpix)...
    return signal

def ADC_converter(signal, resolution):
    digital_signal = []
    resolution_interval = np.linspace(min(signal), max(signal), resolution)
    for i in signal:
        closest_discrete_level = np.abs(resolution_interval - i).argmin()
        digital_signal.append(resolution_interval[closest_discrete_level])
    return digital_signal

def plot_graph(digital_signal, sample_interval, bits_resolution, signal):
    fig, ax = plt.subplots(ncols= 1, nrows= 3, figsize= (13,5))

    for a in ax:
        a.set(xlim = (0, max(sample_interval)), ylim = (min(signal), max(signal)))
        a.set_xlabel('Time[s]', fontsize=15)
        a.set_ylabel('Amplitude', fontsize=15)
        a.grid(True)

    ax[0].plot(x, analog_function_true, color = 'orange')
    ax[0].set_title('Analog signal')

    ax[1].plot(sample_interval, signal, label= "Samples", color='k')
    ax[1].set_title('Samples')

    ax[2].step(sample_interval, digital_signal, linewidth = 2, where='post')
    ax[2].set_title(f"Digital Signal - {bits_resolution} bit resolution", fontsize=18)

    fig.tight_layout()
    plt.show()
    return fig


n = 15
x = np.linspace(0, n, n*25)
frequency_interval = (1, 7, 2)
sample_interval = np.arange(0, n, 0.01) 

bits_resolution = 16
resolution = 2**bits_resolution

analog_function_true = analog_function(x, frequency_interval)
signal = analog_function(sample_interval, frequency_interval)
digital_signal = ADC_converter(signal, resolution)

plot_graph(digital_signal, sample_interval, bits_resolution, signal)

print(analog_function_true)



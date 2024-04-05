import numpy as np
import matplotlib.pyplot as plt

path = r"/home/malum/Downloads/open_data_seg_poland_vibroseis_2ms_1501x282_shot_1.bin"
seismic = np.fromfile(path, dtype= np.float32, count=423282).reshape([1501,282], order = "F")

seismic_trace = seismic[:][100]

sine_frequencies = seismic_trace
sine_amplitudes = np.ones(len(sine_frequencies))

dt = 0.02        # 2ms
total_time = 2.0                      
nt = int(total_time / dt)                    
f_s = 1 / dt     # 50hz

n = np.arange(nt)                   
domain = np.linspace(0, total_time, int(1e6))
discrete_time = n*dt

analog_signal = 0.0
for index, freq in enumerate(sine_frequencies):
    analog_signal += sine_amplitudes[index]*np.sin(2*np.pi*freq*domain)

discrete_sine = 0.0
for index, freq in enumerate(sine_frequencies):
    w_s = 2.0*np.pi*freq/ f_s     
    discrete_sine += sine_amplitudes[index]*np.sin(w_s*n)  

discrete_sine_fft = np.fft.fft(discrete_sine)
discrete_sine_fft *= np.max(sine_amplitudes) / np.max(np.abs(discrete_sine_fft))
f = np.fft.fftfreq(nt, dt)

fig, ax = plt.subplots(ncols=1, nrows=3, figsize=(15,6))

ax[0].plot(domain, analog_signal)
ax[0].stem(discrete_time, discrete_sine, '--k')
ax[0].set_title("Analog Signal", fontsize = 18)
ax[0].set_xlabel("Time [s]", fontsize = 15)

ax[1].plot(n, discrete_sine, '--o')
ax[1].set_xlim([0, nt-1])
ax[1].set_title("Discrete Signal", fontsize = 18)
ax[1].set_xlabel("Discrete Time", fontsize = 15)

ax[2].stem(f, np.abs(discrete_sine_fft))
ax[2].set_ylabel("Amplitudes", fontsize = 15)
ax[2].set_xlabel("Frequencies [Hz]", fontsize = 15)

fig.tight_layout()
plt.grid(axis = 'y')
plt.show()
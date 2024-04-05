import numpy as np
import matplotlib.pyplot as plt

dt = 2e-3
nt = 1501

nx = 282
dx = 25

path = r"/home/malum/Downloads/open_data_seg_poland_vibroseis_2ms_1501x282_shot_1.bin"
seismic = np.fromfile(path, dtype= np.float32, count=nx*nt).reshape([nt,nx], order = "F")

fft_time = np.fft.fft(seismic, axis = 0)
f = np.fft.fftfreq(nt, dt)

fig, ax = plt.subplots(ncols=3, nrows=1)

scale = 0.8*np.std(seismic)
ax[0].set_title("CMP Gather", fontsize = 18)
ax[0].imshow(seismic, cmap='Greys', aspect='auto', vmin=-scale, vmax=scale)

ax[0].set_xticks(np.linspace(0, nx - 1, 5))
ax[0].set_xticklabels(np.linspace(-(nt - 1)*0.5, (nt -1)*0.5, 5))
ax[0].set_xlabel("Offset [m]", fontsize=15)

ax[0].set_yticks(np.linspace(0, nt, 11))
ax[0].set_yticklabels(np.linspace(0, nt-1, 11)*dt)
ax[0].set_ylabel("TWT [s]", fontsize=15)

scale2 = 0.8*np.std(np.abs(fft_time))
ax[1].set_title("FFT of all traces", fontsize=18)
cax = ax[1].imshow(np.abs(fft_time), aspect='auto', vmin=-scale2, vmax=scale2)

ax[1].set_xticks(np.linspace(0, nx - 1, 5))
ax[1].set_xticklabels(np.linspace(-(nt - 1)*0.5, (nt -1)*0.5, 5))
ax[1].set_xlabel("Offset [m]", fontsize=15)

ax[1].set_yticks(np.linspace(0, nt - 1, 8))
#ax[1].set_yticklabels(np.linspace(300, 0, 7))
ax[1].set_yticklabels([0, 125, 212.5, 300, -300, 212.5, 125, 0])
ax[1].set_ylabel("Frequency [Hz]", fontsize=15)
fig.colorbar(cax)

ax[2].set_title("Seismic Trace [100]", fontsize=18)
ax[2].plot(f, np.abs(fft_time[:, 100]))

ax[2].set_xlim([0, 30])
ax[2].set_xlabel("Frequency [Hz]", fontsize=15)
ax[2].set_ylabel("Amplitude", fontsize=15)


mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.show()

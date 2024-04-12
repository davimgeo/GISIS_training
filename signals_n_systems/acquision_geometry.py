import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

path = r"/home/malum/GISIS_training/signals_n_systems/geometry_files"

spread = 96
fixed = 24

total_shots = 251
total_nods = total_shots + spread

stations = np.loadtxt(path + "/stations.txt", comments="#")
shots = np.loadtxt(path + "/shots.txt", comments="#")
relation = np.loadtxt(path + "/relation.txt", comments="#", delimiter=",")

fig, ax = plt.subplots()

total_traces = np.array([])

def init():
    fig.tight_layout()

def update(frame):
    ax.clear()
    global total_traces
    cmp = np.array([])
    for j in range(len(stations)):
        foo = (stations[j] + shots[frame]) * 0.5
        cmp = np.append(cmp, foo)
    total_traces = np.append(total_traces, cmp)   
    
    ax.plot(stations, np.zeros(total_nods), 'vk', label="Receivers")
    ax.plot(shots[frame], 0, 'ob', label="Shots")
    ax.plot(cmp, np.zeros(len(cmp)) - 0.01, 'og', label="CMP")
    ax.set_xlim([min(stations), max(stations)])
    ax.set_ylim([-0.02, 0.02])

    ax.legend(loc="lower left", bbox_to_anchor=(0, 0.02))
    ax.set_title("Acquisition Geometry", fontsize=18)

ani = FuncAnimation(fig, update, frames=len(shots), init_func=init, interval=50, repeat=False)
#ani.save('acquisition_geometry.mp4', writer='ffmpeg')

plt.show()

print(f"There's a total of {len(np.unique(total_traces))} unique traces")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import math
import os
import csv
import statistics


def vec(a):  # calculates vectors
    i_max = np.shape(a)[0]
    v = np.zeros(i_max)
    for i in range(0, i_max-1):
        v[i] = a[i+1]-a[i]  # to next in array
    v[i_max-1] = a[0]-a[i_max-1]  # back to origin
    return v


def min_d(a, v, l, p):  # start point, vector, vector length, data point
    return np.linalg.norm(np.cross((p-a), v))/l


# Curve
N_ref = 10
t = np.linspace(0, 2*math.pi, num=N_ref)
# circle:
r = 1500
c = [500, -500, 1500]
rx = r*np.cos(t)+np.full(N_ref, c[0])
ry = r*np.sin(t)+np.full(N_ref, c[1])
rz = np.full(N_ref, c[2])
vx = vec(rx)
vy = vec(ry)
vz = vec(rz)
l = np.linalg.norm([vx[0], vy[0], vz[0]])

# Measurements
# dataset 1
Start = 750  # 1600  # start frame
N_data = 1000  # 2000 # how many frames
pathToData = sys.argv[1]
meas = (pd.read_csv(pathToData, usecols=["x", "y", "z"], skiprows=Start, nrows=N_data, names=[
        "Frame", "x", "y", "z", "roll", "pitch", "yaw"]).to_numpy() * 1000)
meas_min = np.zeros(N_data)

# #dataset 2
# Start = #1600  # start frame
# N_data = 2000#2000 # how many frames
# pathToData = sys.argv[1]
# meas = (pd.read_csv(pathToData, usecols=["x", "y", "z"], skiprows=Start, nrows=N_data, names=["Frame", "x", "y", "z", "roll", "pitch", "yaw"]).to_numpy()* 1000)
# meas_min=np.zeros(N_data)


for j in range(N_data):
    candidates = np.zeros(N_ref-1)
    for i in range(N_ref-1):
        candidates[i] = min_d(np.array([rx[i], ry[i], rz[i]]), np.array(
            [vx[i], vy[i], vz[i]]), l, np.array([meas[j, 0], meas[j, 1], meas[j, 2]]))
    meas_min[j] = np.amin(candidates)

rms = np.linalg.norm(meas_min)/math.sqrt(N_data)

print(rms)

# #test plot
fig_proj = plt.figure()
proj_plt = fig_proj.add_subplot(111, projection="3d")
proj_plt.quiver(rx, ry, rz, vx, vy, vz, length=1)
proj_plt.scatter(rx, ry, rz, marker="o", s=1, c="black")
proj_plt.scatter(meas[:, 0], meas[:, 1], meas[:, 2], marker="o", s=1, c="red")
proj_plt.set_zlim3d(500, 1500)

plt.show()

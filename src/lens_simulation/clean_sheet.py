import numpy as np
from matplotlib import pyplot as plt
import glob
import json

# Plotting config
plot_cleaned = True

# Read the metadata.json file
metadata = json.load(open('metadata.json'))
diameter =  metadata['sim_parameters']['sim_width']
n_steps = metadata['stages'][-1]['n_steps']
exponent = metadata['lenses'][-1]['exponent']
height = metadata['lenses'][-1]['height']
pixel_size = metadata['sim_parameters']['pixel_size']

print(f'Diameter: {diameter*1e6}um  ')
print(f'Exponent: {exponent}')
print(f'Height: {height*1e6}um')

# Get the start and finish distance of the simulation
paths = glob.glob('1/*mm.npy')
for i, p in enumerate(paths):
    paths[i] = float(p.split('/')[-1].split('m')[0])
finish = max(paths)
start = min(paths)
step_size = (finish-start) / n_steps

# load the simulation array
sim_path = '1/sim.npy'
a = np.load(sim_path)
single = a[:, 0, :]

# Grab the center slice and max value
center_cut = single[:, single.shape[1]//2]
center_max = np.amax(center_cut)

# Threshold the simulation array
top_half = single > center_max/2
cleaned = top_half * single
if plot_cleaned:
    plt.figure()
    plt.imshow(single, aspect='auto', cmap='turbo', extent=([-diameter/2, diameter/2, finish, start]))

    plt.figure()
    plt.imshow(cleaned, aspect='auto', cmap='turbo', extent=([-diameter/2, diameter/2, finish, start]))

# Threshold just the center cut
center_top_half = np.flip(center_cut > center_max/2)

half_width = 0
max_half_width = 0
max_width = 0
max_depth = 0

for i, pixel in enumerate(center_top_half):
    half_width_px = top_half[i][single.shape[1]//2]
    while half_width_px != 0:
        half_width += 1
        half_width_px = top_half[i][single.shape[1]//2 + half_width]

    if half_width * pixel_size > max_half_width:
        max_half_width = half_width * pixel_size
    half_width = 0
max_width = max_half_width * 2

i = 0
while i < len(center_top_half) - 1:
    while center_top_half[i] == 0 and i < len(center_top_half) - 1:
        i += 1
        first = i

    while center_top_half[i] != 0 and i < len(center_top_half) - 1:
        i += 1
        last = i

    depth = (last-first)*step_size
    if depth > max_depth:
        max_depth = depth

print(f'Depth: {max_depth}mm')
print(f'Width: {max_width*1e6}um')

plt.show()



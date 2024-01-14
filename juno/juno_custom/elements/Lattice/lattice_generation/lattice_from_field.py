import pickle
import numpy as np
from matplotlib import pyplot as plt

# load data
with open(r"C:\Users\Dadie1\Downloads\Efield.pkl", "rb") as f:
    ideal_efield = pickle.load(f)

with open(r"C:\Users\Dadie1\Downloads\PlotSLMPhase.pkl", "rb") as f:
    binary_phase = pickle.load(f)

with open(r"C:\Users\Dadie1\Downloads\Maskintensity(1).pkl", "rb") as f:
    pupil_mask = pickle.load(f)

with open(r"C:\Users\Dadie1\Downloads\SampleIntensityAtFocus.pkl", "rb") as f:
    sample_intensity = pickle.load(f)

intensity_efield = np.abs(ideal_efield)**2

# normalise to pi phase
ideal_efield *= np.pi
intensity_efield *= np.pi
binary_phase *= np.pi

# create wavefront
wavefront_field = 1 * np.exp(1j * ideal_efield)
wavefront_field_intensity = 1 * np.exp(1j * intensity_efield)
wavefront_binary = 1 * np.exp(1j * binary_phase)

# find shifted pupils
fft_field = np.fft.fft2(wavefront_field)
fft_field_intensity = np.fft.fft2(wavefront_field_intensity)
fft_binary = np.fft.fft2(wavefront_binary)

# set 0, 0 to 0
fft_field[0, 0] = 0
fft_field_intensity[0, 0] = 0
fft_binary[0, 0] = 0

# inverse shift
fft_field = np.fft.ifftshift(fft_field)
fft_field_intensity = np.fft.ifftshift(fft_field_intensity)
fft_binary = np.fft.ifftshift(fft_binary)

# inverse ffts
ifft_field = np.fft.ifft2(fft_field)
ifft_field_intensity = np.fft.ifft2(fft_field_intensity)
ifft_binary = np.fft.ifft2(fft_binary)

# plot
plt.figure()
plt.imshow(np.abs(ifft_field)**2)
plt.colorbar()

# plt.figure()
# plt.imshow(np.abs(ifft_field_intensity)**2)
# plt.colorbar()

# plt.figure()
# plt.imshow(np.abs(ifft_binary)**2)
# plt.colorbar()
plt.show()

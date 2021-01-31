import numpy as np

data = np.genfromtxt('2.txt')
pixel_height = ((data[:, 0] + 26 - 20)/386).reshape((data.shape[0], 1))
pixel_width = ((data[:, 1] - 56)/386).reshape((data.shape[0], 1))
_id = np.zeros_like(pixel_height)
pixel_temp = np.concatenate((pixel_height, pixel_width), axis=1)
pixel_temp = np.concatenate((_id, pixel_temp), axis=1)


np.savetxt("pixel_pos.csv", pixel_pos, delimiter=',')

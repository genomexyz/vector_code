import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from datetime import datetime, timedelta
from PIL import Image

from PIL import Image
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.image as mpimg

#setting
radar_file = 'radar.png'

# Predefined list of allowed RGB values
allowed_colors = np.array([
    [255, 255, 255],
    [7, 254, 246],
    [0, 150, 255],
    [0, 2, 254],
    [1, 254, 3],
    [0, 199, 3],
    [0, 153, 2],
    [255, 254, 0],
    [255, 200, 1],
    [255, 119, 7],
    [251, 1, 3],
    [201, 0, 2],
    [152, 0, 1],
    [255, 0, 255],
    [152, 0, 254]
])


def find_closest_color(pixel, color_list):
    distances = np.sqrt(np.sum((color_list - pixel) ** 2, axis=1))
    return color_list[np.argmin(distances)]

def find_closest_color_vectorized(pixels, color_list):
    # Expand pixel dimensions to (n_pixels, 1, 3) for broadcasting
    pixels_expanded = pixels[:, np.newaxis, :]
    # Calculate the Euclidean distance between each pixel and each color
    distances = np.sqrt(np.sum((pixels_expanded - color_list) ** 2, axis=2))
    # Find the index of the closest color for each pixel
    closest_color_indices = np.argmin(distances, axis=1)
    # Use the indices to map to the color_list
    closest_colors = color_list[closest_color_indices]
    return closest_colors

def calibrate_color(img3d):
    # Flatten the 3D image array (height x width x 3) into a 2D array (n_pixels x 3)
    pixels = img3d.reshape(-1, 3)
    # Vectorize the closest color function for all pixels
    closest_colors = find_closest_color_vectorized(pixels, allowed_colors)
    #closest_colors = np.array([find_closest_color(pixel, allowed_colors) for pixel in pixels])
    # Reshape the pixel array back to the original image shape
    output_img_array = closest_colors.reshape(img3d.shape)
    return output_img_array.astype(int)

img = Image.open(radar_file).convert("RGB")
img_arr = np.array(img)

# Find the unique colors (unique rows in the pixel array)
pixels = img_arr.reshape(-1, 3)
unique_colors = np.unique(pixels, axis=0)

# Print the unique RGB colors
#print("Unique RGB Colors:")
#for color in unique_colors:
#    print(tuple(color))
#
#exit()


img_arr = calibrate_color(img_arr)

#plt.imshow(img_arr)
#plt.show()
#plt.close()
#

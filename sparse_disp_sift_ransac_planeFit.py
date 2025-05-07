import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load input image
image = cv2.imread('/Users/vedant/Documents/Projects_Flam/stereo_disp/data/min_pattern_floor_left.jpeg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Plane coefficients (from fitting)
a, b, c = 0.0101, -0.0469, 285.7622  # Example values, update with actual ones

# Camera intrinsic parameters (example, replace with actual values)
f_x = 3066  # Example focal length in pixels
f_y = 3066  # Assuming square pixels (same focal length in both x and y directions)
c_x = 239   # Optical center (cx)
c_y = 425   # Optical center (cy)

# Create a grid of pixel coordinates (image coordinates)
height, width = image.shape[:2]
x_range = np.arange(0, width, 1)
y_range = np.arange(0, height, 1)
X, Y = np.meshgrid(x_range, y_range)

# Flatten the grid for easy computation
X_flat = X.flatten()
Y_flat = Y.flatten()

# Calculate the corresponding Z (depth) for each (x, y) coordinate on the plane
Z_flat = a * X_flat + b * Y_flat + c

# Reproject the 3D plane points onto the image plane
u_flat = (f_x * X_flat) / Z_flat + c_x
v_flat = (f_y * Y_flat) / Z_flat + c_y

# Round to nearest integers for pixel coordinates
u = np.round(u_flat).astype(int)
v = np.round(v_flat).astype(int)

# Make sure points are within the image bounds
valid_points = (u >= 0) & (u < width) & (v >= 0) & (v < height)
u_valid = u[valid_points]
v_valid = v[valid_points]

# Project the plane points on the image with a distinct color
for u_proj, v_proj in zip(u_valid, v_valid):
    cv2.circle(image, (u_proj, v_proj), 1, (0, 255, 0), -1)  # Green color for the plane

# Display the image with projected plane
cv2.imshow("Projected Plane on Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally, save the result
cv2.imwrite("projected_plane_on_image.png", image)

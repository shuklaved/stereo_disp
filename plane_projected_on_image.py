import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load image
img = cv2.imread("/Users/vedant/Documents/Projects_Flam/stereo_disp/data/min_pattern_floor_left.jpeg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = img.shape[:2]

# Load depth points
depth_file = "sparse_depths_with_ids_inliers.txt"
points_2D = []
points_depth = []

with open(depth_file, "r") as f:
    next(f)
    for line in f:
        _, x, y, d = line.strip().split(",")
        points_2D.append((float(x), float(y)))
        points_depth.append(float(d))

points_2D = np.array(points_2D)
points_depth = np.array(points_depth)

# Approx camera intrinsics
fx = fy = 3066  # adjust to your camera
cx, cy = w / 2, h / 2

# Convert to 3D
X = (points_2D[:, 0] - cx) * points_depth / fx
Y = (points_2D[:, 1] - cy) * points_depth / fy
Z = points_depth

# Fit plane Z = aX + bY + c
A = np.c_[X, Y, np.ones_like(Z)]
C, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)

# Generate grid in image coords
xx, yy = np.meshgrid(np.linspace(0, w, 100), np.linspace(0, h, 100))
XX = (xx - cx) / fx
YY = (yy - cy) / fy
ZZ = C[0] * XX + C[1] * YY + C[2]

# Project 3D points back to 2D
xx_proj = (XX * ZZ + cx).astype(np.int32)
yy_proj = (YY * ZZ + cy).astype(np.int32)

# Draw plane points on image
img_plane = img_rgb.copy()
for x_p, y_p in zip(xx_proj.flatten(), yy_proj.flatten()):
    if 0 <= x_p < w and 0 <= y_p < h:
        img_plane[y_p, x_p] = [255, 0, 0]  # Red overlay

# Show result
plt.figure(figsize=(12, 6))
plt.imshow(img_plane)
plt.title("Plane Projection Overlay on Image")
plt.axis("off")
plt.show()

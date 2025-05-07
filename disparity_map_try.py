import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Load sparse depth points
depths = []
with open("sparse_depths_with_ids_inliers.txt", "r") as f:
    next(f)
    for line in f:
        _, x, y, depth = line.strip().split(',')
        depths.append((float(x), float(y), float(depth)))

# Convert to numpy arrays
points = np.array([(x, y) for x, y, _ in depths])
values = np.array([depth for _, _, depth in depths])

# Create meshgrid for full image size
img = cv2.imread("/home/vedant/Documents/Projects_Flam/stereo_disp/data/checkerboard_left.jpeg")
h, w = img.shape[:2]
grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

# Interpolate sparse depth map to dense
depth_map = griddata(points, values, (grid_x, grid_y), method='linear')
depth_map[np.isnan(depth_map)] = 0  # Replace NaNs with 0

# Compute gradients
grad_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)
gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

# Compute Laplacian
laplacian = cv2.Laplacian(depth_map, cv2.CV_64F)

# Normalize for visualization
def normalize(img):
    img = img.copy()
    img -= np.min(img)
    img /= (np.max(img) + 1e-8)
    return img

depth_map_vis = normalize(depth_map)
gradient_vis = normalize(gradient_magnitude)
laplacian_vis = normalize(np.abs(laplacian))

# Normalize depth for colormap overlay
depth_colored = cv2.applyColorMap((depth_map_vis * 255).astype(np.uint8), cv2.COLORMAP_JET)

# Resize to match image (in case of mismatch)
if depth_colored.shape[:2] != img.shape[:2]:
    depth_colored = cv2.resize(depth_colored, (w, h))

# Blend with original image (convert to RGB first)
image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
overlay = cv2.addWeighted(image_rgb, 0.6, depth_colored, 0.4, 0)

# Show and save
plt.figure(figsize=(8, 6))
plt.title("Image with Interpolated Depth Overlay")
plt.imshow(overlay)
plt.axis('off')
plt.tight_layout()
plt.show()

# Optional: save the overlay image
cv2.imwrite("overlay_depth_on_image.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

# Plot results
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Interpolated Depth Map")
plt.imshow(depth_map_vis, cmap='jet')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Gradient Magnitude")
plt.imshow(gradient_vis, cmap='hot')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Laplacian")
plt.imshow(laplacian_vis, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

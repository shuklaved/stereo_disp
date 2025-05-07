import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# Load input image
image = cv2.imread('/Users/vedant/Documents/Projects_Flam/stereo_disp/data/min_pattern_floor_left.jpeg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load depth points
depths = []
with open("sparse_depths_with_ids_inliers.txt", "r") as f:
    next(f)
    for line in f:
        _, x, y, depth = line.strip().split(',')
        depths.append((float(x), float(y), float(depth)))

# Normalize depths
depth_values = [d[2] for d in depths]
depth_min, depth_max = min(depth_values), max(depth_values)
norm = plt.Normalize(vmin=depth_min, vmax=depth_max)
colormap = plt.get_cmap('jet')

# Track text positions using KDTree to avoid overlap
text_positions = []
tree = None
min_dist = 15  # Minimum distance between texts

for x, y, depth in depths:
    color = colormap(norm(depth))
    color_bgr = tuple(int(255 * c) for c in color[:3][::-1])

    # Draw circle
    cv2.circle(image, (int(x), int(y)), 3, color_bgr, -1)

    # Avoid overlapping text
    if tree:
        dists, _ = tree.query([(x, y)], k=1)
        if dists[0] < min_dist:
            continue  # too close to previous text
    text = f"{depth:.2f}"
    cv2.putText(image, text, (int(x) + 5, int(y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_bgr, 1, cv2.LINE_AA)
    text_positions.append((x, y))
    tree = KDTree(text_positions)

# Save and show
cv2.imwrite("projected_depth_with_text.png", image)
cv2.imshow("Projected Depth with Text", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# Load grayscale stereo images
imgR = cv2.imread('/Users/vedant/Documents/Projects_Flam/stereo_disp/data/near_left.jpg', cv2.IMREAD_GRAYSCALE)
imgL = cv2.imread('/Users/vedant/Documents/Projects_Flam/stereo_disp/data/near_right.jpg', cv2.IMREAD_GRAYSCALE)

# Resize images to (width=1280, height=960)
target_size = (1080, 1920)
imgL = cv2.resize(imgL, target_size, interpolation=cv2.INTER_AREA)
imgR = cv2.resize(imgR, target_size, interpolation=cv2.INTER_AREA)

# imgR = cv2.imread('/home/vedant/Documents/Projects_Flam/stereo_disp/data/small_mov_left.jpeg', cv2.IMREAD_GRAYSCALE)
# imgL = cv2.imread('/home/vedant/Documents/Projects_Flam/stereo_disp/data/small_mov_right.jpeg', cv2.IMREAD_GRAYSCALE)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors for both images
kpL, desL = sift.detectAndCompute(imgL, None)
kpR, desR = sift.detectAndCompute(imgR, None)

# Create BFMatcher with L2 norm for SIFT descriptors
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Match descriptors
matches = bf.match(desL, desR)

# Sort matches by distance (lower is better)
matches = sorted(matches, key=lambda x: x.distance)

# Extract matched points from keypoints
ptsL = np.array([kpL[m.queryIdx].pt for m in matches])
ptsR = np.array([kpR[m.trainIdx].pt for m in matches])

# Use RANSAC to remove outliers and compute the fundamental matrix
F, mask = cv2.findFundamentalMat(ptsL, ptsR, cv2.FM_RANSAC, 0.5, 0.85)

# Mask the good matches (inliers)
inliers = mask.ravel() == 1
ptsL_inliers = ptsL[inliers]
ptsR_inliers = ptsR[inliers]

# Visualize matches (only inliers)
img_matches = cv2.drawMatches(imgL, kpL, imgR, kpR, [matches[i] for i in range(len(matches)) if inliers[i]], None, flags=2)

# Save the image showing inlier matches
cv2.imwrite("sparse_disparity_matches_inliers.png", img_matches)

# Print disparities for inliers
sparse_disparities = []
for i, (ptL, ptR) in enumerate(zip(ptsL_inliers, ptsR_inliers)):
    disparity = ptL[0] - ptR[0]  # xL - xR
    sparse_disparities.append((ptL, disparity))
    print(f"Point {i}: Location={ptL}, Disparity={disparity:.2f}")

baseline_cm = 7.9 # baseline in cm calculated using the sensor data
focal_length_mm = 4.745 # Focal length in mm
sensor_size_x_mm = 6.4 # Camera sensor size in 
sensor_size_y_mm = 4.8
image_res_x = 4000
image_res_y = 3000

focal_length_pxs = focal_length_mm * (image_res_x/sensor_size_x_mm)
print('Focal_Length_in mm: ',focal_length_pxs)

sparse_depths = []
for i, (ptL, ptR) in enumerate(zip(ptsL_inliers, ptsR_inliers)):
    disparity = ptL[0] - ptR[0]  # xL - xR
    if disparity > 0:
        depth = (focal_length_pxs * baseline_cm) / disparity  # Assuming a baseline of 1 unit
        sparse_depths.append((ptL, depth))
        print(f"Point {i}: Location={ptL}, Depth={depth:.2f}")

# Save sparse disparities to a file
with open("sparse_disparities_with_ids_inliers.txt", "w") as f:
    f.write("Index, X, Y, Disparity\n")
    for i, (pt, disp) in enumerate(sparse_disparities):
        x, y = pt
        f.write(f"{i}, {x:.2f}, {y:.2f}, {disp:.2f}\n")

# Save sparse depths to a file 
with open("sparse_depths_with_ids_inliers.txt", "w") as f:
    f.write("Index, X, Y, Depth\n")
    for i, (pt, depth) in enumerate(sparse_depths):
        x, y = pt
        f.write(f"{i}, {x:.2f}, {y:.2f}, {depth:.2f}\n")

# For putting depth text on the dominant image
imgL = cv2.imread('/Users/vedant/Documents/Projects_Flam/stereo_disp/data/floor_near_left.jpg')
target_size = (1080, 1920)
imgL = cv2.resize(imgL, target_size, interpolation=cv2.INTER_AREA)

image_rgb = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)

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
    cv2.circle(imgL, (int(x), int(y)), 3, color_bgr, -1)

    # Avoid overlapping text
    if tree:
        dists, _ = tree.query([(x, y)], k=1)
        if dists[0] < min_dist:
            continue  # too close to previous text
    text = f"{depth:.2f}"
    cv2.putText(imgL, text, (int(x) + 5, int(y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_bgr, 1, cv2.LINE_AA)
    text_positions.append((x, y))
    tree = KDTree(text_positions)

# Save and show
cv2.imwrite("projected_depth_with_text.png", imgL)
cv2.imshow("Projected Depth with Text", imgL)
cv2.waitKey(0)
cv2.destroyAllWindows()
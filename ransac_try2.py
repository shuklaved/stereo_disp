import cv2
import numpy as np

# Load grayscale stereo images
imgR = cv2.imread('/Users/vedant/Documents/Projects_Flam/stereo_disp/data/floor_left.jpg', cv2.IMREAD_GRAYSCALE)
imgL = cv2.imread('/Users/vedant/Documents/Projects_Flam/stereo_disp/data/floor_right.jpg', cv2.IMREAD_GRAYSCALE)

# Resize images to (width=1280, height=960)
target_size = (1080, 1920)
imgL = cv2.resize(imgL, target_size, interpolation=cv2.INTER_AREA)
imgR = cv2.resize(imgR, target_size, interpolation=cv2.INTER_AREA)

# imgR = cv2.imread('/home/vedant/Documents/Projects_Flam/stereo_disp/data/min_pattern_floor_left.jpeg', cv2.IMREAD_GRAYSCALE)
# imgL = cv2.imread('/home/vedant/Documents/Projects_Flam/stereo_disp/data/min_pattern_floor_right.jpeg', cv2.IMREAD_GRAYSCALE)

num = 500 # Number of matches (max)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors for both images
kpL, desL = sift.detectAndCompute(imgL, None)
kpR, desR = sift.detectAndCompute(imgR, None)

# Match descriptors using Brute-Force Matcher with L2 distance for SIFT
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(desL, desR)
matches = sorted(matches, key=lambda x: x.distance)

# Extract matched keypoints
ptsL = np.float32([kpL[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
ptsR = np.float32([kpR[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# Use RANSAC to find inlier matches based on the fundamental matrix
F, mask = cv2.findFundamentalMat(ptsL, ptsR, cv2.FM_RANSAC, 1.0, 0.99)

# Select inlier matches
inlier_matches = [m for i, m in enumerate(matches) if mask[i]]

# Optionally limit to best inliers (for display / sanity)
inlier_matches = sorted(inlier_matches, key=lambda x: x.distance)[:num]

# Draw matches and compute disparity at each match point
sparse_disparities = []
for m in inlier_matches[:num]:  # adjust count as needed
    ptL = kpL[m.queryIdx].pt
    ptR = kpR[m.trainIdx].pt
    disparity = ptL[0] - ptR[0]  # xL - xR
    sparse_disparities.append((ptL, disparity))

# Copy original image to draw on
annotated_img = cv2.cvtColor(imgL.copy(), cv2.COLOR_GRAY2BGR)

# Annotate with alternate offsets to reduce overlap
for i, (pt, disp) in enumerate(sparse_disparities[:num]):
    x, y = int(pt[0]), int(pt[1])
    offset_y = -10 if i % 2 == 0 else 10  # alternate up/down
    cv2.circle(annotated_img, (x, y), 3, (0, 255, 0), -1)
    cv2.putText(annotated_img, str(i), (x + 5, y + offset_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)


cv2.imwrite("annotated_left_with_ids.png", annotated_img)

# Visualize matches (optional)
img_matches = cv2.drawMatches(imgL, kpL, imgR, kpR, inlier_matches[:num], None, flags=2)

# Sensor data for depth estimation
baseline_cm = 8 # baseline in cm calculated using the sensor data
#focal_length_mm = 4.745 # Focal length in mm
#sensor_size_x_mm = 6.4 # Camera sensor size in mm
#sensor_size_y_mm = 4.8
#image_res_x = 4000
#image_res_y = 3000

#focal_length_pxs = focal_length_mm * (image_res_x/sensor_size_x_mm)
focal_length_pxs = 4095
#print('Focal_Length_in mm: ',focal_length_pxs)

sparse_depths = []
for i, (pt, disp) in enumerate(sparse_disparities):
    if disp > 1:  # Avoid division by zero or very small disparity
        depth = (focal_length_pxs * baseline_cm) / disp
        sparse_depths.append((pt, depth))
        print(f"Point {i}: Location={pt}, Depth={depth:.2f} cm")

# Save sparse depths to a file 
with open("sparse_depths_with_ids_inliers.txt", "w") as f:
    f.write("Index, X, Y, Depth\n")
    for i, (pt, depth) in enumerate(sparse_depths):
        x, y = pt
        f.write(f"{i}, {x:.2f}, {y:.2f}, {depth:.2f}\n")

# Print disparities
for i, (pt, disp) in enumerate(sparse_disparities):
    print(f"Point {i}: Location={pt}, Disparity={disp:.2f}")

cv2.imwrite("sparse_disparity_matches_new.png", img_matches)

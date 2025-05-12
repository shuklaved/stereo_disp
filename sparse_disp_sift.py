import cv2
import numpy as np

# Load grayscale stereo images
imgR = cv2.imread('/Users/vedant/Documents/Projects_Flam/stereo_disp/data/near_left.jpeg', cv2.IMREAD_GRAYSCALE)
imgL = cv2.imread('/Users/vedant/Documents/Projects_Flam/stereo_disp/data/near_right.jpeg', cv2.IMREAD_GRAYSCALE)

# Resize images to (width=1280, height=960)
target_size = (960, 1280)
imgL = cv2.resize(imgL, target_size, interpolation=cv2.INTER_AREA)
imgR = cv2.resize(imgR, target_size, interpolation=cv2.INTER_AREA)

# imgR = cv2.imread('/home/vedant/Documents/Projects_Flam/stereo_disp/data/min_pattern_floor_left.jpeg', cv2.IMREAD_GRAYSCALE)
# imgL = cv2.imread('/home/vedant/Documents/Projects_Flam/stereo_disp/data/min_pattern_floor_right.jpeg', cv2.IMREAD_GRAYSCALE)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors for both images
kpL, desL = sift.detectAndCompute(imgL, None)
kpR, desR = sift.detectAndCompute(imgR, None)

# Match descriptors using Brute-Force Matcher with L2 distance for SIFT
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(desL, desR)
matches = sorted(matches, key=lambda x: x.distance)

# Draw matches and compute disparity at each match point
sparse_disparities = []
for m in matches[:10]:  # limit to 50 best matches for simplicity
    ptL = kpL[m.queryIdx].pt
    ptR = kpR[m.trainIdx].pt
    disparity = ptL[0] - ptR[0]  # xL - xR
    sparse_disparities.append((ptL, disparity))

# Copy original image to draw on
annotated_img = cv2.cvtColor(imgL.copy(), cv2.COLOR_GRAY2BGR)

# Annotate with alternate offsets to reduce overlap
for i, (pt, disp) in enumerate(sparse_disparities[:10]):
    x, y = int(pt[0]), int(pt[1])
    offset_y = -10 if i % 2 == 0 else 10  # alternate up/down
    cv2.circle(annotated_img, (x, y), 3, (0, 255, 0), -1)
    cv2.putText(annotated_img, str(i), (x + 5, y + offset_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)


cv2.imwrite("annotated_left_with_ids.png", annotated_img)

# Visualize matches (optional)
img_matches = cv2.drawMatches(imgL, kpL, imgR, kpR, matches[:10], None, flags=2)
# cv2.imshow("Matches", img_matches)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Print disparities
for i, (pt, disp) in enumerate(sparse_disparities):
    print(f"Point {i}: Location={pt}, Disparity={disp:.2f}")

cv2.imwrite("sparse_disparity_matches.png", img_matches)

# Save sparse disparities to a file
with open("sparse_disparities_with_ids.txt", "w") as f:
    f.write("Index, X, Y, Disparity\n")
    for i, (pt, disp) in enumerate(sparse_disparities[:100]):
        x, y = pt
        f.write(f"{i}, {x:.2f}, {y:.2f}, {disp:.2f}\n")

import cv2

# Load grayscale stereo image pair
imgL = cv2.imread('/home/vedant/Documents/Projects_Flam/stereo_disp/im0.png', cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread('/home/vedant/Documents/Projects_Flam/stereo_disp/im1.png', cv2.IMREAD_GRAYSCALE)

# Create StereoBM object
stereo_BM = cv2.StereoBM_create(numDisparities=16*6, blockSize=5)

# Create StereoSGBM object
stereo_SGBM = cv2.StereoSGBM_create(
    minDisparity=50,
    numDisparities=16*2,       # Must be divisible by 16
    blockSize=5,
    P1=4 * 3 * 5**2,
    P2=32 * 3 * 5**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

disp_SGBM = stereo_SGBM.compute(imgL, imgR).astype('float32') / 16.0

# Normalize for visualization
disp_SGBM = cv2.normalize(disp_SGBM, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
cv2.imwrite("disparity_sgbm.png", disp_SGBM)

# Compute disparity map
disp_BM = stereo_BM.compute(imgL, imgR)

# Normalize for visualization
disp_BM = cv2.normalize(disp_BM, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
cv2.imwrite("disparity_bm.png", disp_BM)


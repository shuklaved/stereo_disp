import cv2
import numpy as np

boxes = []
drawing = False
start_point = None

def mouse_draw(event, x, y, flags, param):
    global drawing, start_point, boxes, current_image

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        img_copy = current_image.copy()
        cv2.rectangle(img_copy, start_point, (x, y), (0, 255, 0), 2)
        cv2.imshow(param, img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        boxes.append((start_point, end_point))
        cv2.rectangle(current_image, start_point, end_point, (0, 255, 0), 2)
        cv2.imshow(param, current_image)

def get_center(pt1, pt2):
    return ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)

# --- For Image 1 ---
image1 = cv2.imread('/Users/vedant/Documents/Projects_Flam/stereo_disp/data/far_right.jpg')
current_image = image1.copy()
cv2.namedWindow("Image 1")
cv2.setMouseCallback("Image 1", mouse_draw, "Image 1")
cv2.imshow("Image 1", current_image)
cv2.waitKey(0)
cv2.destroyWindow("Image 1")

# --- For Image 2 ---
image2 = cv2.imread('/Users/vedant/Documents/Projects_Flam/stereo_disp/data/far_left.jpg')
current_image = image2.copy()
cv2.namedWindow("Image 2")
cv2.setMouseCallback("Image 2", mouse_draw, "Image 2")
cv2.imshow("Image 2", current_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Compute centers and distance
if len(boxes) == 2:
    c1 = get_center(*boxes[0])
    c2 = get_center(*boxes[1])
    print("Center 1:", c1)
    print("Center 2:", c2)
    dist = np.linalg.norm(np.array(c1) - np.array(c2))
    print("Euclidean Distance:", dist)
else:
    print("You need to draw one box per image.")

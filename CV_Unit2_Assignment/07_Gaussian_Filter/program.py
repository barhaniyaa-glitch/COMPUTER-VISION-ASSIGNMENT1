import cv2

img = cv2.imread("input.jpg")

smoothed = cv2.GaussianBlur(img, (5, 5), 0)

cv2.imwrite("output.png", smoothed)

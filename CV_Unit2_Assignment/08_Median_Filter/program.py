import cv2

img = cv2.imread("input.jpg")

filtered = cv2.medianBlur(img, 5)

cv2.imwrite("output.png", filtered)

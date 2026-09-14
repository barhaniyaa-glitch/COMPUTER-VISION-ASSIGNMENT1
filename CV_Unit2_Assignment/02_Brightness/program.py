import cv2
import numpy as np

img = cv2.imread("input.jpg")

brightness_value = 80

enhanced = cv2.add(img, np.full(img.shape, brightness_value, dtype=np.uint8))

sample_y, sample_x = 100, 100
print("Pixel before enhancement:", img[sample_y, sample_x])
print("Pixel after enhancement:", enhanced[sample_y, sample_x])

cv2.imwrite("output.png", enhanced)

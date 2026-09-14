import cv2
import numpy as np

img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
img_float = np.float32(img)

dft = cv2.dft(img_float, flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shifted = np.fft.fftshift(dft)

print("Original image shape:", img.shape)
print("DFT result shape:", dft.shape)
print("Shifted DFT shape:", dft_shifted.shape)

magnitude = cv2.magnitude(dft_shifted[:, :, 0], dft_shifted[:, :, 1])
magnitude_log = 20 * np.log(magnitude + 1)
magnitude_norm = cv2.normalize(magnitude_log, None, 0, 255, cv2.NORM_MINMAX)

cv2.imwrite("output.png", magnitude_norm.astype(np.uint8))

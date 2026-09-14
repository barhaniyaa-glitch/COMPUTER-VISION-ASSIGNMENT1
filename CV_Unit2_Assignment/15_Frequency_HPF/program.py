import cv2
import numpy as np

img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
img_float = np.float32(img)

dft = cv2.dft(img_float, flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shifted = np.fft.fftshift(dft)

rows, cols = img.shape
crow, ccol = rows // 2, cols // 2

radius = 30
mask = np.ones((rows, cols, 2), np.float32)
cv2.circle(mask, (ccol, crow), radius, (0, 0), -1)

filtered_dft = dft_shifted * mask

dft_inverse_shifted = np.fft.ifftshift(filtered_dft)
img_reconstructed = cv2.idft(dft_inverse_shifted)
img_reconstructed = cv2.magnitude(img_reconstructed[:, :, 0], img_reconstructed[:, :, 1])

img_reconstructed = cv2.normalize(img_reconstructed, None, 0, 255, cv2.NORM_MINMAX)

cv2.imwrite("output.png", img_reconstructed.astype(np.uint8))

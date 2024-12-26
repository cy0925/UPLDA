import cv2
import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift

# 读取图像
image = cv2.imread('./1.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
image = image.astype(np.float32) / 255.0

# 提取V通道（亮度）
v_channel = image[:,:,2]

# Step 1: 对数变换
log_v = np.log(v_channel+1e-6)

# Step 2: FFT
fft_v = fft2(log_v)
fft_v_shifted = fftshift(fft_v)

# Step 3: 高通滤波
rows, cols = v_channel.shape
crow, ccol = rows // 2 , cols // 2
d = 30  # 高通滤波器的大小
mask = np.ones((rows, cols), float)
# mask[crow-d:crow+d, ccol-d:ccol+d] = 0
mask *= 0.45
mask[0,0] = 0.35

filtered_fft_v = fft_v_shifted * mask

# Step 4: IFFT
ifft_v_shifted = ifftshift(filtered_fft_v)
ifft_v = ifft2(ifft_v_shifted)
ifft_v = np.real(ifft_v)

# Step 5: 指数变换
exp_v = np.exp(ifft_v)
exp_v = np.clip(exp_v, 0, 1)

# 将处理后的V通道回插入HSV图像
image[:,:,2] = exp_v

# 转回RGB
enhanced_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

# 显示和保存图像
# cv2.imshow('Original Image', cv2.imread('/mnt/data/image.png'))
# cv2.imshow('Enhanced Image', enhanced_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cv2.imwrite('enhanced_image.png', enhanced_image)

import numpy as np
import cv2
import objective_function as obf
import matplotlib.pyplot as plt
import evaluate as ev
import gradient_descent as gd
import image_proposing as imp

# 导入图片
img_initial = cv2.imread('./image.png',cv2.IMREAD_GRAYSCALE)

# 向图片加入高斯噪声
img = imp.img_guassian(img_initial)

img_initial = imp.img_normalization(img_initial)

# 归一化处理
u = imp.img_normalization(img)

cv2.imshow('noisy image', u)
cv2.waitKey(0)
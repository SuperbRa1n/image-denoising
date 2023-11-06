import numpy as np
import cv2
import objective_function as obf
import matplotlib.pyplot as plt
import evaluate as ev
import gradient_descent as gd
import image_proposing as imp

# 导入图片
img = cv2.imread("./image.png", cv2.IMREAD_GRAYSCALE)

# 向图片加入高斯噪声
img = imp.img_guassian(img, 0, 20)

# 归一化处理
u = imp.img_normalization(img)

# 梯度下降
X = 2 * u
lam = 1
count = 100
initial_step = 1


def f(X):
    return obf.f(X, u, lam)


def diff_f(X):
    return obf.diff_f(X, u, lam)


X, loss, norm_gradient = gd.gd_bb(f, diff_f, X, initial_step, count)

X = imp.img_normalization(X)  # 归一化处理

# 输出psnr
print(ev.psnr(X, u, 1))

# 绘制图像
plt.figure(1)
plt.plot(range(count), np.log10(loss))
plt.xlabel("Iteration")
plt.ylabel("value of the loss function(log)")
plt.figure(2)
plt.plot(range(count), np.log10(norm_gradient))
plt.xlabel("Iteration")
plt.ylabel("norm of the gradient(log)")
plt.show()

# 显示图片结果
cv2.imshow("BB_step", X)
cv2.waitKey(0)

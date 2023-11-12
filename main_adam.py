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
print(ev.psnr(u, ev.img_initial, 255))
# 梯度下降
X = np.ones_like(img) * np.random.rand()
lam = 1
count = 200
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8


def f(X):
    return obf.f(X, u, lam)


def diff_f(X):
    return obf.diff_f(X, u, lam)


X, loss, norm_gradient, psnr = gd.gd_adam(f, diff_f, X, beta1, beta2, epsilon, count)

X = imp.img_normalization(X)  # 归一化处理

print(ev.psnr(X, img_initial, 255))

# 绘制图像
plt.figure(1)
plt.plot(range(count),np.log10(loss))
plt.xlabel('Iteration')
plt.ylabel('value of the loss function(log)')
plt.figure(2)
plt.plot(range(count),np.log10(norm_gradient))
plt.xlabel('Iteration')
plt.ylabel('norm of the gradient(log)')
plt.figure(3)
plt.plot(range(count), psnr)
plt.xlabel('Iteration')
plt.ylabel('psnr')
plt.show()

# 显示图片结果
cv2.imshow("Adam_step", X)
cv2.waitKey(0)
import numpy as np
import cv2 
import objective_function as obf
import matplotlib.pyplot as plt
import evaluate as ev
import gradient_descent as gd
import image_proposing as imp

# 导入图片
img = cv2.imread('./image.png',cv2.IMREAD_GRAYSCALE)

# 向图片加入高斯噪声
img = imp.img_guassian(img)

# 归一化处理
u = imp.img_normalization(img)

# 创建部分缺失的观测图像
missing_ratio = 0.5 # 缺失比例
missing_mask = np.random.normal(0, 1, u.shape) > missing_ratio

u_observed = u * missing_mask

# 梯度下降
X = np.ones_like(img) * np.random.rand()
lam = 0.001
step = 0.01
count = 1000

def regularizer(X: np.matrix) -> float:
    alpha = 0.5
    l1_term = alpha * np.sum(np.abs(X))
    l2_term = (1 - alpha) * np.sum(X**2)
    return l1_term + l2_term

def diff_regularizer(X: np.matrix) -> np.matrix:
    alpha = 0.5
    return alpha * np.sign(X) + (1 - alpha) * 2 * X

def f(X: np.matrix) -> float:
    return obf.f_optimized(X, u, missing_mask, lam, regularizer)
def diff_f(X: np.matrix) -> np.matrix:
    return obf.diff_f_optimized(X, u, missing_mask, lam, diff_regularizer)

X,loss,norm_gradient,psnr = gd.gd(f,diff_f,X,step,count)

X = imp.img_normalization(X) # 归一化处理

# 输出psnr
print(ev.psnr(X, ev.img_initial, 255))

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
cv2.imshow('ElasticNet',X)
cv2.waitKey(0)

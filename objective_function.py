import numpy as np
import numpy as np
from scipy.signal import convolve2d


# 水平向前差分矩阵
def Mat_dx(X:np.matrix):
    return np.mat(X[1:,:] - X[:-1,:])

# 竖直向前差分矩阵
def Mat_dy(X:np.matrix):
    return np.mat(X[:,1:] - X[:,:-1])

# Laplace算子(差分的导数)
def Mat_laplacian(X: np.matrix):
    # 定义离散拉普拉斯算子
    laplacian_operator = np.array([[0, 1, 0],
                              [1, -4, 1],
                              [0, 1, 0]])
    return convolve2d(X, laplacian_operator, mode='same', boundary='wrap')


# X为真实图片，Y为带噪声的图像
def f(X:np.matrix, Y:np.matrix, lam:float):
    return 0.5*np.linalg.norm(X-Y)**2+lam*(np.linalg.norm(Mat_dx(X))**2+np.linalg.norm(Mat_dy(X))**2)

# f的导数
def diff_f(X:np.matrix, Y:np.matrix, lam:float):
    return X - Y - 2*lam*Mat_laplacian(X)

if __name__ == '__main__':
    X = np.matrix([[1,2,3],[4,5,6],[7,8,9]])
    print(np.diff(X,axis=1))
    print(np.diff(X,axis=0))
    print(Mat_laplacian(X))


import numpy as np
from scipy.signal import convolve2d

# 创建一个示例矩阵
matrix = np.array([[1, 2, 3,4],
                  [4, 5, 6,9],
                  [7, 8, 9,4]])

# 定义离散拉普拉斯算子
laplacian_operator = np.array([[0, 1, 0],
                              [1, -4, 1],
                              [0, 1, 0]])

# 使用 convolve2d 函数计算矩阵的离散化二阶导数
laplacian = convolve2d(matrix, laplacian_operator, mode='same', boundary='wrap')

print("矩阵的离散化二阶导数 (拉普拉斯结果):")
print(laplacian)

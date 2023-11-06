import numpy as np
import numpy as np
from scipy.signal import convolve2d


# 水平向前差分矩阵
def Mat_dx(X: np.matrix):
    """
    Calculate the difference between consecutive rows of a matrix.

    Args:
        X: The input matrix.

    Returns:
        A matrix containing the differences between consecutive rows of the input matrix.
    """
    return np.mat(X[1:, :] - X[:-1, :])


# 竖直向前差分矩阵
def Mat_dy(X: np.matrix):
    """
    Calculate the difference between adjacent columns of a matrix.

    Args:
        X: A numpy matrix.

    Returns:
        A numpy matrix representing the difference between adjacent columns of X.
    """
    return np.mat(X[:, 1:] - X[:, :-1])


# Laplace算子(差分的导数)
def Mat_laplacian(X: np.matrix):
    """
    Apply the discrete Laplacian operator to a matrix.

    Args:
        X: A numpy matrix.

    Returns:
        A numpy matrix representing the result of convolving X with the discrete Laplacian operator.
    """
    # 定义离散拉普拉斯算子
    laplacian_operator = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    return convolve2d(X, laplacian_operator, mode="same", boundary="wrap")


# X为真实图片，Y为带噪声的图像
def f(X: np.matrix, Y: np.matrix, lam: float):
    """
    Calculate the objective function value.

    Args:
        X: A numpy matrix.
        Y: A numpy matrix.
        lam: A float representing the regularization parameter.

    Returns:
        The objective function value calculated as 0.5 * ||X - Y||^2 + lam * (||Mat_dx(X)||^2 + ||Mat_dy(X)||^2).
    """
    return 0.5 * np.linalg.norm(X - Y) ** 2 + lam * (
        np.linalg.norm(Mat_dx(X)) ** 2 + np.linalg.norm(Mat_dy(X)) ** 2
    )


# f的导数
def diff_f(X: np.matrix, Y: np.matrix, lam: float):
    """
    Calculate the difference between two matrices with a regularization term.

    Args:
        X: A numpy matrix.
        Y: A numpy matrix.
        lam: A float representing the regularization parameter.

    Returns:
        A numpy matrix representing the difference between X and Y, subtracted by 2 times the regularization term calculated using Mat_laplacian(X).
    """
    return X - Y - 2 * lam * Mat_laplacian(X)


if __name__ == "__main__":
    X = np.matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(np.diff(X, axis=1))
    print(np.diff(X, axis=0))
    print(Mat_laplacian(X))

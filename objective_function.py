import numpy as np
import numpy as np
from scipy.signal import convolve2d
from typing import Callable
# 水平向前差分矩阵
def Mat_dx(X: np.matrix) -> np.matrix:
    """
    Calculate the difference between consecutive rows of a matrix.

    Args:
        X: The input matrix.

    Returns:
        A matrix containing the differences between consecutive rows of the input matrix.
    """
    return np.mat(X[1:, :] - X[:-1, :])


# 竖直向前差分矩阵
def Mat_dy(X: np.matrix) -> np.matrix:
    """
    Calculate the difference between adjacent columns of a matrix.

    Args:
        X: A numpy matrix.

    Returns:
        A numpy matrix representing the difference between adjacent columns of X.
    """
    return np.mat(X[:, 1:] - X[:, :-1])


# Laplace算子(差分的导数)
def Mat_laplacian(X: np.matrix) -> np.matrix:
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
def f(X: np.matrix, Y: np.matrix, lam: float) -> float:
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
def diff_f(X: np.matrix, Y: np.matrix, lam: float) -> np.matrix:
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


# 另外一个目标函数
def f_optimized(X:np.matrix, Y: np.matrix, O:np.matrix, lam: float, regularizer: Callable[...,float]) -> float:
    '''
    The `f_optimized` function calculates the optimized objective function value.

    Args:
        X: A numpy matrix representing the input variable.
        Y: A numpy matrix representing the target variable.
        O: A numpy matrix representing the observation matrix.
        lam: A float representing the regularization parameter.
        regularizer: A callable function that computes the regularization term.

    Returns:
        A float representing the optimized objective function value.
    '''
    return 0.5 * np.linalg.norm(Y - O * X)**2 + lam * regularizer(X)

# 函数的导数
def diff_f_optimized(X:np.matrix, Y: np.matrix, O:np.matrix, lam: float, diff_regularizer: Callable[...,np.matrix]) -> np.matrix:
    '''
    The `diff_f_optimized` function calculates the optimized objective function value with a differential regularizer.

    Args:
        X: A numpy matrix representing the input variable.
        Y: A numpy matrix representing the target variable.
        O: A numpy matrix representing the observation matrix.
        lam: A float representing the regularization parameter.
        diff_regularizer: A callable function that computes the differential regularization term.

    Returns:
        A numpy matrix representing the optimized objective function value.
    '''
    return O * (O * X - Y) - lam * diff_regularizer(X)
import numpy as np
import evaluate as ev
import cv2


def gd(func, gradient, x, step, count):
    """
    Performs gradient descent optimization.

    Args:
        func (callable): The objective function to be minimized.
        gradient (callable): The gradient function of the objective function.
        x (numpy.ndarray): The initial value of the optimization variable.
        step (float): The step size for each iteration of gradient descent.
        count (int): The number of iterations to perform.

    Returns:
        tuple: A tuple containing the optimized variable, the list of losses at each iteration,
            the list of norm of gradients at each iteration, and the list of PSNR values at each iteration.
    """
    loss = []
    norm_gradient = []
    psnr = []
    for i in range(count):
        x -= step * gradient(x)
        loss.append(func(x))
        norm_gradient.append(np.linalg.norm(gradient(x)))
        psnr.append(ev.psnr(x, ev.img_initial, 255))
        print(f"第{i}步:loss={loss[i]}")
    return x, loss, norm_gradient, psnr


def gd_bb(func, gradient, x, initial_step, count):
    """
    Performs gradient descent optimization with Barzilai-Borwein step size.

    Args:
        func (callable): The objective function to be minimized.
        gradient (callable): The gradient function of the objective function.
        x (numpy.ndarray): The initial value of the optimization variable.
        initial_step (float): The initial step size for the first iteration of gradient descent.
        count (int): The number of iterations to perform.

    Returns:
        tuple: A tuple containing the optimized variable, the list of losses at each iteration,
            the list of norm of gradients at each iteration, and the list of PSNR values at each iteration.
    """
    loss = []
    norm_gradient = []
    psnr = []
    for i in range(count):
        g = gradient(x)
        if i>0:
            delta_g: np.matrix = g - g_old
            delta_x: np.matrix = x - x_old
            step = np.dot(delta_x.flatten(), delta_x.flatten()) / np.dot(delta_g.flatten(), delta_x.flatten())
        else:
            step = initial_step
        g_old = g.copy()
        x_old = x.copy()
        loss.append(func(x))
        norm_gradient.append(np.linalg.norm(g))
        psnr.append(ev.psnr(x, ev.img_initial, 255))
        x -= step*g
        if np.all(x-x_old<1e-7):
            for j in range(i+1,count):
                loss.append(func(x))
                norm_gradient.append(np.linalg.norm(g))
                psnr.append(ev.psnr(x, ev.img_initial, 255))
            break
        print(f"第{i}步:loss={loss[i]}")
    return x, loss, norm_gradient, psnr

def gd_adam(func, gradient, x, beta1, beta2, epsilon, count):
    """
    Performs gradient descent optimization with Adam optimizer.

    Args:
        func (callable): The objective function to be minimized.
        gradient (callable): The gradient function of the objective function.
        x (numpy.ndarray): The initial value of the optimization variable.
        beta1 (float): The exponential decay rate for the first moment estimates.
        beta2 (float): The exponential decay rate for the second moment estimates.
        epsilon (float): A small value added to the denominator for numerical stability.
        count (int): The number of iterations to perform.

    Returns:
        tuple: A tuple containing the optimized variable, the list of losses at each iteration,
            the list of norm of gradients at each iteration, and the list of PSNR values at each iteration.
"""
    m = 0
    v = 0
    loss_arr = []
    nog_arr = []
    psnr = []
    for i in range(count):
        t = i + 1
        loss = func(x)
        grad = gradient(x)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad**2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        x -= m_hat / (np.sqrt(np.linalg.norm(v_hat)) + epsilon)
        loss_arr.append(loss)
        nog_arr.append(np.linalg.norm(grad))
        psnr.append(ev.psnr(x, ev.img_initial, 255))
        print(f"第{i}步:loss={loss_arr[i]}")
    return x, loss_arr, nog_arr, psnr
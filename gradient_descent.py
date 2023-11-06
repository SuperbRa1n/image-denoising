import numpy as np


def gd(func, gradient, x, step, count):
    """
    Perform gradient descent optimization on a given function.

    Args:
        func: The objective function to be minimized.
        gradient: The gradient function of the objective function.
        x: The initial value of the optimization variable.
        step: The step size for each iteration of gradient descent.
        count: The number of iterations to perform.

    Returns:
        A tuple containing the optimized variable, a list of loss values at each iteration, and a list of norm of gradients at each iteration.
    """
    loss = []
    norm_gradient = []
    for i in range(count):
        x -= step * gradient(x)
        loss.append(func(x))
        norm_gradient.append(np.linalg.norm(gradient(x)))
        print(f"第{i}步:loss={loss[i]}")
    return x, loss, norm_gradient


def gd_bb(func, gradient, x, initial_step, count):
    """
    Perform gradient descent optimization using Barzilai-Borwein step size.

    Args:
        func (function): A function that calculates the loss value.
        gradient (function): A function that calculates the gradient of the loss function.
        x (np.matrix): A numpy matrix representing the initial point.
        initial_step (float): The initial step size for the optimization.
        count (int): The number of iterations.

    Returns:
        tuple: A tuple containing the final point, a list of loss values at each iteration, and a list of the norm of gradients at each iteration.

    Note:
        The step size is updated using the Barzilai-Borwein method.
    """

    loss = []
    norm_gradient = []
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
        x -= step*g
        if np.all(x-x_old<1e-7):
            for j in range(i+1,count):
                loss.append(func(x))
                norm_gradient.append(np.linalg.norm(g))
            break
        print(f"第{i}步:loss={loss[i]}")
    return x, loss, norm_gradient
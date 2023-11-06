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
    return x, loss, norm_gradient


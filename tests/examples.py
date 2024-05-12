import numpy as np

def quadratic_function(x, Q, need_hessian=False):
    """Compute the function value, gradient, and Hessian for a quadratic form f(x) = x^T Q x."""
    f = x.T @ Q @ x
    g = 2 * Q @ x
    h = 2 * Q if need_hessian else None
    return f, g, h

def example_1(x, need_hessian=False):
    Q = np.array([[1, 0],
                  [0, 1]])  # Contour lines are circles
    return quadratic_function(x, Q, need_hessian)

def example_2(x, need_hessian=False):
    Q = np.array([[1, 0],
                  [0, 100]])  # Contour lines are axis aligned ellipses
    return quadratic_function(x, Q, need_hessian)

def example_3(x, need_hessian=False):
    R = np.array([[np.sqrt(3)/2, -0.5],
                  [0.5, np.sqrt(3)/2]])
    D = np.array([[100, 0],
                  [0, 1]])
    Q = R.T @ D @ R  # Compute the matrix Q as the product of R^T, D, and R
    return quadratic_function(x, Q, need_hessian)

def rosenbrock_function(x, need_hessian=False):
    """ Compute Rosenbrock's function, its gradient, and optionally its Hessian. """
    f = 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
    g = np.array([-400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0]),
                  200 * (x[1] - x[0]**2)])


    if need_hessian:
        h = np.array([[1200 * x[0]**2 - 400 * x[1] + 2, -400 * x[0]],
                      [-400 * x[0], 200]])
    else:
        h = None
    return f, g, h

def linear_function(x, need_hessian=False):
    """ Compute a linear function f(x) = a^T x, its gradient, and Hessian (if needed). """
    a = np.array([2, 3])  # Example vector a
    f = np.dot(a, x)
    g = a
    h = np.zeros((2, 2)) if need_hessian else None
    return f, g, h

def exponential_function(x, need_hessian=False):
    """ Compute the given exponential function and its derivatives. """
    x1, x2 = x
    f = np.exp(x1 + 3*x2 - 0.1) + np.exp(x1 - 3*x2 - 0.1) + np.exp(-x1 - 0.1)
    g = np.array([np.exp(x1 + 3*x2 - 0.1) + np.exp(x1 - 3*x2 - 0.1) - np.exp(-x1 - 0.1),
                  3*np.exp(x1 + 3*x2 - 0.1) - 3*np.exp(x1 - 3*x2 - 0.1)])
    if need_hessian:
        h = np.array([[np.exp(x1 + 3*x2 - 0.1) + np.exp(x1 - 3*x2 - 0.1) + np.exp(-x1 - 0.1),
                       3*np.exp(x1 + 3*x2 - 0.1) - 3*np.exp(x1 - 3*x2 - 0.1)],
                      [3*np.exp(x1 + 3*x2 - 0.1) - 3*np.exp(x1 - 3*x2 - 0.1),
                       9*np.exp(x1 + 3*x2 - 0.1) + 9*np.exp(x1 - 3*x2 - 0.1)]])
    else:
        h = None
    return f, g, h
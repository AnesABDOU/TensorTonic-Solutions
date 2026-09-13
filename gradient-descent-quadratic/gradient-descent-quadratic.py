def gradient_descent_quadratic(a: float, b: float, c: float, x0: float, lr: float, steps: int) -> float:
    """
    Returns the final scalar x after the requested iterations.
    """
    # grad_quad_x = 2*a*x + b
    x = x0
    for i in range(steps):
        grad_x = 2*a*x + b
        x = x - lr*grad_x
    return x
    pass
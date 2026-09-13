import numpy as np

def adam_step(
    param: list,
    grad: list,
    m: list,
    v: list,
    t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (param_new, m_new, v_new) as NumPy arrays.
    """
    # Write code here
    m = np.array(m)
    v = np.array(v)
    grad = np.array(grad)
    m = beta1*m+(1-beta1)*grad
    v = beta2*v+(1-beta2)*grad**2
    mt_hat = m/(1-beta1**t)
    vt_hat = v/(1-beta2**t)
    theta = param-lr*mt_hat/(np.sqrt(vt_hat) + eps)
    return theta, m, v
    pass
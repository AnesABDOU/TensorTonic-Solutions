import numpy as np

def dropout(
    x: list,
    p: float = 0.5,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (output, dropout_pattern) as NumPy arrays matching the shape of x.
    """
    x_arr = np.array(x)
    shape = x_arr.shape
    
    if rng is None:
        rand_vals = np.random.random(shape)
    else:
        rand_vals = rng.random(shape)

    mask = rand_vals < 1-p
    
    output = np.array(shape)
    dropout_pattern = np.array(shape)

    dropout_pattern = 1/(1-p) * mask
            
    output = dropout_pattern * x_arr

    return output, dropout_pattern
    pass
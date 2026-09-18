import numpy as np

def apply_homogeneous_transform(T: list, points: list) -> np.ndarray:
    """
    Returns transformed points with shape (3,) or (N, 3).
    """
    pts = np.asarray(points)
    T = np.asarray(T)

    if pts.ndim == 1:
        pts_h = np.append(pts, 1)
        return np.dot(T, pts_h.T)[:3]
        
    one = np.ones((pts.shape[0], 1))
    pts_h = np.hstack([pts, one])
    res = np.dot(T, pts_h.T).T[:, :3]

    return res
    
    pass
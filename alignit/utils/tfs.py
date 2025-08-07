import numpy as np
import transforms3d as t3d


def are_tfs_close(a, b=None, lin_tol=1e-2, ang_tol=1e-2):
    """
    Check if two transformation matrices are close to each other within specified tolerances.

    Parameters:
        a (numpy.ndarray): The first transformation matrix.
        b (numpy.ndarray, optional): The second transformation matrix. If not provided, it defaults to the identity matrix.
        lin_tol (float, optional): The linear tolerance for closeness. Defaults to 1e-9.
        ang_tol (float, optional): The angular tolerance for closeness. Defaults to 1e-9.

    Returns:
        bool: True if the matrices are close, False otherwise.
    """
    if b is None:
        b = np.eye(4)
    d = np.linalg.inv(a) @ b
    if not np.allclose(d[:3, 3], np.zeros(3), atol=lin_tol):
        return False
    yaw = np.arctan2(d[1, 0], d[0, 0])
    pitch = np.arcsin(-d[2, 0])
    roll = np.arctan2(d[2, 1], d[2, 2])
    rpy = np.array([roll, pitch, yaw])
    return np.allclose(rpy, np.zeros(3), atol=ang_tol)


def get_pose_str(pose, degrees=True):
    xyz = pose[:3, 3]
    # Ensure data is float64 to avoid NumPy 2.0 copy=False error inside transforms3d
    rpy = t3d.euler.mat2euler(np.asarray(pose[:3, :3], dtype=np.float64), axes="sxyz")
    if degrees:
        rpy = np.rad2deg(rpy)
    return f"{xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}, {rpy[0]:.3f}, {rpy[1]:.3f}, {rpy[2]:.3f}"


def print_pose(pose):
    printable = get_pose_str(pose, degrees=True)
    print(f"Pose (xyzrpy, deg): {printable}")

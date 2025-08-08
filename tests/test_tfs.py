import numpy as np

from alignit.utils.tfs import are_tfs_close, get_pose_str


def test_are_tfs_close_identity():
    I = np.eye(4)
    assert are_tfs_close(I, I)
    assert are_tfs_close(I)


def test_are_tfs_close_tolerances():
    A = np.eye(4)
    B = np.eye(4)
    B[:3, 3] = [1e-3, 0, 0]
    assert are_tfs_close(A, B, lin_tol=1e-2)
    assert not are_tfs_close(A, B, lin_tol=1e-4)


def test_get_pose_str_format():
    pose = np.eye(4)
    s = get_pose_str(pose)
    assert isinstance(s, str)
    parts = s.split(',')
    assert len(parts) == 6

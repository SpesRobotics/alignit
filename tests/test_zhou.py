import numpy as np

from alignit.utils.zhou import se3_sixd, sixd_se3


def test_identity_roundtrip():
    M = np.eye(4, dtype=np.float32)
    v = se3_sixd(M)
    M2 = sixd_se3(v)
    assert M2.shape == (4, 4)
    assert np.allclose(M2, M, atol=1e-6)


def test_random_rotations_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(5):
        A = rng.normal(size=(3, 3))
        Q, _ = np.linalg.qr(A)
        if np.linalg.det(Q) < 0:
            Q[:, 0] = -Q[:, 0]
        t = rng.normal(size=3)
        M = np.eye(4, dtype=np.float32)
        M[:3, :3] = Q
        M[:3, 3] = t
        v = se3_sixd(M)
        M2 = sixd_se3(v)
        assert np.allclose(M2, M, atol=1e-5)

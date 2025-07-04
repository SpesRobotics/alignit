# Reference: https://arxiv.org/abs/1812.07035

import numpy as np


def se3_sixd(m):
    rot_mat = m[:3, :3]
    xyz = m[:3, 3]
    rot_vec = rot_mat[:, :2].reshape(6)
    return np.concatenate([xyz, rot_vec], axis=0).astype(np.float32)


def sixd_se3(vec):
    assert vec.shape[-1] == 9, "Input vector must have length 9"
    trans = vec[:3]
    rot6 = vec[3:9]
    # Reconstruct the first two columns of the rotation matrix
    col1 = rot6[[0, 2, 4]]
    col2 = rot6[[1, 3, 5]]
    # Orthonormalize
    b1 = col1 / np.linalg.norm(col1)
    b2 = col2 - np.dot(b1, col2) * b1
    if np.linalg.norm(b2) < 1e-6:
        fallback = np.array([1, 0, 0], dtype=np.float32)
        b2 = (
            fallback
            if not np.allclose(b1, fallback)
            else np.array([0, 1, 0], dtype=np.float32)
        )
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    rot_mat = np.stack((b1, b2, b3), axis=-1)

    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = rot_mat
    M[:3, 3] = trans
    return M


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Identity test
    M = np.eye(4, dtype=np.float32)
    vec = se3_sixd(M)
    M_rec = sixd_se3(vec)
    assert np.allclose(M_rec, M, atol=1e-6), "Identity test failed"

    # Random transforms
    for _ in range(10):
        A = rng.normal(size=(3, 3))
        Q, _ = np.linalg.qr(A)
        if np.linalg.det(Q) < 0:
            Q[:, 0] = -Q[:, 0]
        t = rng.normal(size=3)
        M = np.eye(4, dtype=np.float32)
        M[:3, :3] = Q
        M[:3, 3] = t
        vec = se3_sixd(M)
        M_rec = sixd_se3(vec)
        assert np.allclose(M_rec, M, atol=1e-5), "Random transform failed"

    # Collinearity fallback
    M = np.eye(4, dtype=np.float32)
    M[:3, :2] = np.array([[1, 2], [0, 0], [0, 0]], dtype=np.float32)
    vec = se3_sixd(M)
    M_rec = sixd_se3(vec)
    assert M_rec.shape == (4, 4), "Collinearity fallback failed"

    print("All tests passed!")

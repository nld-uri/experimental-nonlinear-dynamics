"""Check every number in figures/SOD.tikz, and the identities it asserts.

The figure builds a GSVD of the pair (A, B) by whitening B in the SVD basis
of A. This script (a) reproduces the figure's geometry from its three inputs,
(b) tests the two factorizations the figure claims, and (c) checks that the
resulting Phi and generalized singular values agree with python/gsvd.py on
random pairs. Run from python/.

Convention used by the figure: the shaded ellipse for a matrix M is
E(M) = M^T(unit ball), so its semi-axes are the singular values of M along
the right singular vectors of M. Right multiplication acts on the set from
the left: E(MQ) = Q^T E(M) and E(MD) = D E(M). A LEFT orthogonal factor of M
is invisible in E(M) -- which is why the row-4 panels looked right while
carrying the wrong matrix.
"""
import numpy as np
from gsvd import gsvd

THETA = 30.0                       # degrees, the figure's theta
S1D = np.array([3.0, 1.0])         # Sigma_1 in drawing units (x20 pt)
BAX, BANG = np.array([50.0, 30.0]), -30.0   # E(B): 50pt @ -30 deg, 30pt @ 60


def rot(deg):
    t = np.deg2rad(deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def polar(v):
    return float(np.hypot(*v)), float(np.rad2deg(np.arctan2(v[1], v[0])))


def figure_numbers():
    """Every length and angle the .tikz file hard-codes."""
    V1 = rot(THETA)
    S1 = np.diag(S1D*20)
    A = S1 @ V1.T                                   # U_1 = I; only E(.) is drawn
    B = np.diag(BAX) @ rot(BANG).T
    B1 = B @ V1 @ np.linalg.inv(S1)
    U2, s2, V2t = np.linalg.svd(B1)
    V2 = V2t.T
    for i in range(2):                              # sign convention of the figure
        if V2[0, i] < 0:
            V2[:, i] *= -1
            U2[:, i] *= -1
    out = {}
    out["v2 angles"] = [polar(V2[:, i])[1] for i in range(2)]
    out["sigma2"] = s2
    r = 20.0                                        # radius of the U_1 circle, pt
    for i, nm in enumerate(("21", "22")):
        v = r*V2[:, i]
        Sv = np.diag(S1D) @ v
        VSv = rot(THETA) @ Sv
        out[f"row4 p1 v{nm}"] = polar(v)
        out[f"row4 p2 S1 v{nm}"] = polar(Sv)
        out[f"row4 p3 phi{i+1}"] = polar(VSv)
        out[f"row3 p1 s2 v{nm}"] = polar(s2[i]*v)
        out[f"row3 p2 s2 S1 v{nm}"] = polar(s2[i]*Sv)
        out[f"row3 p3 s2 phi{i+1}"] = polar(s2[i]*VSv)
    # E(B_1): the row-2 p3 and row-3 p1 ellipse
    out["E(B1) axes"] = [(s2[i]*r, polar(V2[:, i])[1]) for i in range(2)]
    return out


def identities(A, B):
    """The two factorizations. Returns (figure's claim, correct claim)."""
    U1, s1, V1t = np.linalg.svd(A, full_matrices=False)
    V1, S1 = V1t.T, np.diag(s1)
    B1 = B @ V1 @ np.linalg.inv(S1)
    U2, s2, V2t = np.linalg.svd(B1, full_matrices=False)
    V2 = V2t.T
    Phi = V1 @ S1 @ V2
    A1 = U1 @ V2
    return {
        "B  = U2 S2 Phi^T          ": np.allclose(B, U2 @ np.diag(s2) @ Phi.T),
        "A  = U1 Phi^T   (WRONG)   ": np.allclose(A, U1 @ Phi.T),
        "A  = A1 S1 V1^T (WRONG)   ": np.allclose(A, A1 @ S1 @ V1.T),
        "A  = A1 Phi^T,  A1 = U1V2 ": np.allclose(A, A1 @ Phi.T),
        "A1 orthogonal             ": np.allclose(A1.T @ A1, np.eye(A1.shape[1])),
    }, Phi, s2


def main():
    print("Figure geometry (lengths in pt, angles in degrees)")
    for k, v in figure_numbers().items():
        print(f"  {k:22s} {v}")

    print("\nIdentities on the figure's own 2x2 pair")
    V1 = rot(THETA)
    A = np.diag(S1D*20) @ V1.T
    B = np.diag(BAX) @ rot(BANG).T
    res, _, _ = identities(A, B)
    for k, v in res.items():
        print(f"  {k} {v}")

    print("\nIdentities and agreement with gsvd.py on random pairs")
    rng = np.random.default_rng(0)
    for t in range(200):
        n = rng.integers(2, 6)
        A = rng.normal(size=(int(n)+3, int(n)))
        B = rng.normal(size=(int(n)+4, int(n)))
        res, Phi, s2 = identities(A, B)
        assert res["B  = U2 S2 Phi^T          "]
        assert res["A  = A1 Phi^T,  A1 = U1V2 "]
        assert res["A1 orthogonal             "]
        assert not res["A  = U1 Phi^T   (WRONG)   "]
        _, _, X, c, s = gsvd(A, B)
        # the figure's pair is (I, Sigma_2), so its generalized values are 1/sigma2
        assert np.allclose(np.sort(c/s), np.sort(1.0/s2)), t
        # Phi and X span the same directions column by column
        Pn = Phi/np.linalg.norm(Phi, axis=0)
        Xn = X/np.linalg.norm(X, axis=0)
        M = np.abs(Pn.T @ Xn)
        assert np.allclose(np.sort(M.max(axis=1)), 1.0), t
    print("  200 random pairs: B and A factorizations, generalized values c/s")
    print("  equal to 1/sigma_2, and Phi parallel to gsvd's X, column by column.")


if __name__ == "__main__":
    main()

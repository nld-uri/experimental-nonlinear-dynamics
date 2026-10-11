"""Generalized singular value decomposition, for the SOD-based routines.

SciPy has no gsvd, so it is built here the standard way -- a QR of the
stacked pair followed by a cosine-sine decomposition (Van Loan 1976,
Paige and Saunders 1981).

Given A (p x n) and B (q x n) with [A; B] of full column rank,

    M = [A; B] = Q R,        Q = [Q1; Q2] with orthonormal columns,
    Q1 = U C Z',  Q2 = V S Z',   C'C + S'S = I,

so with X' = Z' R,

    A = U C X',   B = V S X',

which is MATLAB's convention for [U,V,X,C,S] = gsvd(A,B,0).

The CS decomposition is taken from one SVD of Q1; the sines are then the
column norms of Q2 Z, computed directly rather than as sqrt(1 - c^2), so
they stay accurate where c is close to one -- which is exactly the
regime the smooth modes live in.

One deliberate difference from MATLAB: the pairs are returned with the
generalized singular values c/s in DESCENDING order, so column 0 is the
smoothest mode. MATLAB returns them ascending, which is why sodfilt.m
has to fliplr its factors and this module's callers do not.

Copyright by David Chelidze. Python port.
"""
import numpy as np


def gsvd(A, B):
    """Return U, V, X, c, s with A = U diag(c) X.T and B = V diag(s) X.T.

    A : (p, n) array
    B : (q, n) array
    Both must have n columns and [A; B] must have rank n.
    c/s comes back in descending order; c**2 + s**2 == 1.
    """
    A = np.atleast_2d(np.asarray(A, float))
    B = np.atleast_2d(np.asarray(B, float))
    p, n = A.shape
    q, nb = B.shape
    if nb != n:
        raise ValueError("A and B must have the same number of columns")
    M = np.vstack([A, B])
    if np.linalg.matrix_rank(M) < n:
        raise np.linalg.LinAlgError(
            "[A; B] is column rank deficient, so the generalized singular "
            "values are not defined. Lengthen the strand, lower the "
            "embedding dimension, or regularize the data.")
    Q, R = np.linalg.qr(M)                     # economy: Q is (p+q, n)
    Q1, Q2 = Q[:p, :], Q[p:, :]
    U, c, Zt = np.linalg.svd(Q1, full_matrices=False)   # c descending
    Z = Zt.T
    T = Q2 @ Z                                 # columns are orthogonal
    s = np.linalg.norm(T, axis=0)              # accurate for small s
    V = np.zeros_like(T)
    nz = s > 0
    V[:, nz] = T[:, nz]/s[nz]
    X = R.T @ Z                                # A = U diag(c) X.T
    return U, V, X, c, s


def sod(S, Sdot=None):
    """Smooth orthogonal decomposition of a strand.

    S    : (d, N) strand, one column per point. The mean is NOT removed
           here -- remove it before calling, as the MATLAB code does.
    Sdot : (d, N-1) or (d, N) data-based derivative. Defaults to the
           first difference along the strand, which is what sodfilt uses.

    Returns Phi (SOMs, columns), Psi (SPMs, columns), lam (SOVs), and Q
    (SOCs, columns), all ordered smoothest first, so that

        S ~= Phi[:, :k] @ Psi[:, :k].T @ S

    is the k-dimensional smoothest approximation.
    """
    S = np.atleast_2d(np.asarray(S, float))
    if Sdot is None:
        Sdot = np.diff(S, axis=1)
    U, V, X, c, s = gsvd(S.T, np.asarray(Sdot, float).T)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = np.where(s > 0, (c/np.where(s > 0, s, 1.0))**2, np.inf)
    Phi = X                                     # SOMs
    Psi = np.linalg.inv(X.T)                    # SPMs
    Qc = U*c                                    # SOCs
    return Phi, Psi, lam, Qc


def smooth_project(S, k, Sdot=None):
    """The k-dimensional smoothest approximation to the strand S."""
    Phi, Psi, _, _ = sod(S, Sdot)
    return Phi[:, :k] @ (Psi[:, :k].T @ S)

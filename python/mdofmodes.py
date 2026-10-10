"""Modal parameters and FRF matrix of a linear M-C-K system.

Python counterpart of mdofmodes.m (Appendix A). Chapter 19.

    w_p, z_p, U, H = mdofmodes(M, C, K, w)

M, C, K : n x n mass, damping and stiffness matrices (C may be zero)
w       : frequency grid in rad/s (any length nw), may be empty

w_p : undamped natural frequencies, ascending (rad/s)
z_p : modal damping ratios
U   : n x n mode matrix. If C is proportional (U^T C U diagonal to
      1e-8 relative), U holds the real mass-normalized modes,
      U^T M U = I. Otherwise U holds the complex modes, one per
      conjugate pair, scaled so the largest entry of each is 1.
H   : nw x n x n receptance matrix, H[k] = (K - w^2 M + i w C)^-1,
      computed from the modal/residue expansion (not by inversion).
"""
import numpy as np
from scipy.linalg import eig, eigh


def mdofmodes(M, C, K, w=()):
    M, C, K = (np.asarray(a, float) for a in (M, C, K))
    n = M.shape[0]
    w = np.atleast_1d(np.asarray(w, float))
    # undamped problem: K U = M U Omega^2, U^T M U = I (eigh does this)
    lam2, U = eigh(K, M)
    w_p = np.sqrt(lam2)
    U = U*np.sign(U[0] + (U[0] == 0))       # first nonzero entry positive
    Cm = U.T @ C @ U
    offdiag = np.abs(Cm - np.diag(np.diag(Cm))).max()
    proportional = offdiag <= 1e-8*max(np.abs(Cm).max(), 1e-300)
    H = np.zeros((w.size, n, n), complex)
    if proportional:
        z_p = np.diag(Cm)/(2*w_p)
        for p in range(n):
            den = w_p[p]**2 - w**2 + 2j*z_p[p]*w_p[p]*w
            H += np.outer(U[:, p], U[:, p])[None]/den[:, None, None]
        return w_p, z_p, U, H
    # general viscous damping: 2n first-order state space
    A = np.block([[np.zeros((n, n)), np.eye(n)],
                  [-np.linalg.solve(M, K), -np.linalg.solve(M, C)]])
    lam, V = eig(A)
    for r in range(2*n):                       # residue form of the FRF
        phi, lr = V[:n, r], lam[r]
        a_r = phi @ (2*lr*M + C) @ phi           # modal 'a' constant
        H += np.outer(phi, phi)[None]/(a_r*(1j*w - lr))[:, None, None]
    keep = lam.imag > 0                         # one of each conjugate pair
    lam, Phi = lam[keep], V[:n, keep]
    order = np.argsort(np.abs(lam))
    lam, Phi = lam[order], Phi[:, order]
    w_p, z_p = np.abs(lam), -lam.real/np.abs(lam)
    j = np.argmax(np.abs(Phi), axis=0)          # largest entry -> real, = 1
    Phi = Phi/Phi[j, np.arange(Phi.shape[1])]
    return w_p, z_p, Phi, H

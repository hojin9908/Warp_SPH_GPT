import math

import warp as wp

from input.struct_eisph import EISPHptl


# Exclude the subject point from gradient and Laplacian sums.
R2_EPS = wp.constant(1.0e-12)


@wp.func
def Kernel_w_Wendland_2d(r: float, h: float) -> float:
    """
    Calculate the two-dimensional Wendland C2 kernel value [1/m^2].

    r: distance between two EISPH points [m]
    h: smoothing length [m]

    return: normalized kernel value for support radius 2h
    """
    q = r / h
    value = float(0.0)
    if q < 2.0:
        u = 1.0 - 0.5 * q
        value = 7.0 * u * u * u * u * (1.0 + 2.0 * q) \
            / (4.0 * math.pi * h * h)
    return value


@wp.func
def Kernel_dw_Wendland_2d(r: float, h: float) -> float:
    """
    Calculate dW/dr of the two-dimensional Wendland C2 kernel [1/m^3].

    r: distance between two EISPH points [m]
    h: smoothing length [m]

    return: radial derivative dW/dr; zero outside support 2h
    """
    q = r / h
    value = float(0.0)
    if q < 2.0:
        u = 1.0 - 0.5 * q
        value = -35.0 * q * u * u * u / (4.0 * math.pi * h * h * h)
    return value


@wp.func
def _gradient(rij: wp.vec3, r2: float, h: float) -> wp.vec3:
    """Return the uncorrected gradient grad_i W_ij in the x-z plane [1/m^3]."""
    r = wp.sqrt(r2)
    return Kernel_dw_Wendland_2d(r, h) * rij / r


@wp.func
def _correct_gradient(correction: wp.mat22, grad_w: wp.vec3) -> wp.vec3:
    """Apply the subject point's 2x2 KGC matrix to an x-z kernel gradient."""
    return wp.vec3(correction[0, 0] * grad_w[0] + correction[0, 1] * grad_w[2],
                   0.0,
                   correction[1, 0] * grad_w[0] + correction[1, 1] * grad_w[2])


@wp.func
def _laplace_weight(volume: float, rij: wp.vec3, grad_w: wp.vec3,
                    r2: float) -> float:
    """Return the signed PDF coefficient A_ij for the Laplacian [1/m^2]."""
    return 2.0 * volume * wp.dot(rij, grad_w) / r2


@wp.kernel
def Kernel_prepare_kgc(P_sph: EISPHptl,
                       P_bnd: EISPHptl,
                       grid_sph: wp.uint64,
                       grid_bnd: wp.uint64,
                       support: float,
                       h: float) -> None:
    """
    Precompute the two-dimensional kernel-gradient correction (KGC).

    P_sph, P_bnd: fixed fluid and ghost EISPH particle structures
    grid_sph, grid_bnd: HashGrid handles built from the fixed positions
    support: neighbour-query radius 2h [m]
    h: smoothing length [m]

    Index convention: i / j are fluid points and bj is a ghost neighbour.

    # Output
    P_sph.kgc[i], the inverse first-order moment matrix in x-z.
    Singular moment matrices fall back to the identity matrix.
    """
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid_sph, tid)
    ri = P_sph.pos[i]
    m00 = float(0.0)
    m01 = float(0.0)
    m10 = float(0.0)
    m11 = float(0.0)
    support2 = support * support

    # EISPH - EISPH contribution to -sum(V_j * r_ij outer grad_i W_ij).
    for j in wp.hash_grid_query(grid_sph, ri, support):
        rij = ri - P_sph.pos[j]
        r2 = wp.dot(rij, rij)
        if r2 > R2_EPS and r2 < support2:
            grad_w = _gradient(rij, r2, h)
            volume = P_sph.m[j] / P_sph.rho[j]
            m00 = m00 - volume * rij[0] * grad_w[0]
            m01 = m01 - volume * rij[0] * grad_w[2]
            m10 = m10 - volume * rij[2] * grad_w[0]
            m11 = m11 - volume * rij[2] * grad_w[2]

    # EISPH - Bnd contribution completes support near all four walls.
    for bj in wp.hash_grid_query(grid_bnd, ri, support):
        ribj = ri - P_bnd.pos[bj]
        r2 = wp.dot(ribj, ribj)
        if r2 > R2_EPS and r2 < support2:
            grad_w = _gradient(ribj, r2, h)
            volume = P_bnd.m[bj] / P_bnd.rho[bj]
            m00 = m00 - volume * ribj[0] * grad_w[0]
            m01 = m01 - volume * ribj[0] * grad_w[2]
            m10 = m10 - volume * ribj[2] * grad_w[0]
            m11 = m11 - volume * ribj[2] * grad_w[2]

    # Invert the 2x2 moment matrix once because the Eulerian geometry is fixed.
    determinant = m00 * m11 - m01 * m10
    correction = wp.mat22(1.0, 0.0, 0.0, 1.0)
    if wp.abs(determinant) > 1.0e-8:
        inverse_det = 1.0 / determinant
        correction = wp.mat22(m11 * inverse_det, -m01 * inverse_det,
                              -m10 * inverse_det, m00 * inverse_det)
    P_sph.kgc[i] = correction

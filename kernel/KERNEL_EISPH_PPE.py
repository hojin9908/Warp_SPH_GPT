import warp as wp

from input.struct_eisph import EISPHptl
from kernel.KERNEL_EISPH_KNL import (
    R2_EPS,
    _correct_gradient,
    _gradient,
    _laplace_weight,
)


@wp.kernel
def Kernel_build_ppe_eisph(P_sph: EISPHptl,
                           P_bnd: EISPHptl,
                           grid_sph: wp.uint64,
                           grid_bnd: wp.uint64,
                           support: float,
                           h: float,
                           rho0: float,
                           dt: float) -> None:
    """
    Build the one-pass implicit pressure-Poisson terms.

    P_sph, P_bnd: fluid and ghost EISPH particle structures
    grid_sph, grid_bnd: fixed-position HashGrid handles
    support: neighbour-query radius 2h [m]
    h: smoothing length [m]
    rho0: reference fluid density [kg/m^3]
    dt: time interval [s]

    Index convention: i / j are fluid subject / neighbour and bj is a ghost
    neighbour. The PDF Laplacian convention uses normally negative A_ij and
    A_i*p_i-(Ap)_i=rho0*div(u*)/dt.

    The neighbour-pressure term reads p^t only. A separate solve kernel writes
    p^(t+1), so all subjects use the same previous pressure time level.

    # Output
    P_sph.divergence[i], P_sph.Aij[i], P_sph.bi[i], P_sph.Aijpj[i]
    """
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid_sph, tid)
    ri = P_sph.pos[i]
    vi = P_sph.vel_star[i]
    correction = P_sph.kgc[i]
    div_u = float(0.0)
    Aij = float(0.0)
    Aijpj = float(0.0)
    support2 = support * support

    # EISPH - EISPH divergence, sum(Aij) and sum(Aij*p_j^t).
    for j in wp.hash_grid_query(grid_sph, ri, support):
        rij = ri - P_sph.pos[j]
        r2 = wp.dot(rij, rij)
        if r2 > R2_EPS and r2 < support2:
            grad_w = _correct_gradient(correction, _gradient(rij, r2, h))
            volume = P_sph.m[j] / P_sph.rho[j]
            div_u = div_u + volume * wp.dot(P_sph.vel_star[j] - vi, grad_w)
            coefficient = _laplace_weight(volume, rij, grad_w, r2)
            Aij = Aij + coefficient
            Aijpj = Aijpj + coefficient * P_sph.pres[j]

    # EISPH - Bnd uses predicted ghost velocity and mirrored pressure p^t.
    for bj in wp.hash_grid_query(grid_bnd, ri, support):
        ribj = ri - P_bnd.pos[bj]
        r2 = wp.dot(ribj, ribj)
        if r2 > R2_EPS and r2 < support2:
            grad_w = _correct_gradient(correction, _gradient(ribj, r2, h))
            volume = P_bnd.m[bj] / P_bnd.rho[bj]
            div_u = div_u + volume * wp.dot(P_bnd.vel_star[bj] - vi, grad_w)
            coefficient = _laplace_weight(volume, ribj, grad_w, r2)
            Aij = Aij + coefficient
            j = P_bnd.mirror[bj]
            Aijpj = Aijpj + coefficient * P_sph.pres[j]

    P_sph.divergence[i] = div_u
    P_sph.Aij[i] = Aij
    P_sph.bi[i] = rho0 * div_u / dt
    P_sph.Aijpj[i] = Aijpj


@wp.kernel
def Kernel_remove_bi_mean(P_sph: EISPHptl, count: int) -> None:
    """
    Enforce compatibility of the closed-cavity Neumann PPE.

    P_sph: fluid EISPH particles containing bi and its sum in residual[0]
    count: number of fluid points [N_sph]

    # Output
    P_sph.bi[i] has zero arithmetic mean. The sum was calculated by the
    parallel Warp array reduction before this per-particle subtraction.
    """
    i = wp.tid()
    mean = P_sph.residual[0] / float(count)
    P_sph.bi[i] = P_sph.bi[i] - mean


@wp.kernel
def Kernel_solve_ppe_eisph(P_sph: EISPHptl) -> None:
    """
    Update pressure once with the implicit diagonal relation.

    P_sph: fluid EISPH particles containing Aij, bi and Aijpj

    The off-diagonal contribution was frozen at p^t during assembly. Every
    active row evaluates hat(p)_i = (bi_i + Aijpj_i) / Aij_i; an inactive
    row retains its previous pressure. The kernel stores
    p_i^(t+1) = hat(p)_i - hat(p)_0. The common shift fixes the pressure gauge
    without skipping an active PPE row.

    # Output
    P_sph.pres[i]
    """
    i = wp.tid()
    gauge = float(0.0)
    gauge_Aij = P_sph.Aij[0]
    if wp.abs(gauge_Aij) > 1.0e-12:
        gauge = (P_sph.bi[0] + P_sph.Aijpj[0]) / gauge_Aij

    pressure = P_sph.pres[i]
    Aij = P_sph.Aij[i]
    if wp.abs(Aij) > 1.0e-12:
        pressure = (P_sph.bi[i] + P_sph.Aijpj[i]) / Aij
    P_sph.pres[i] = pressure - gauge

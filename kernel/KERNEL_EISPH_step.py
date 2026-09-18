import warp as wp

from input.struct_eisph import EISPHptl
from kernel.KERNEL_EISPH_KNL import (
    R2_EPS,
    _correct_gradient,
    _gradient,
    _laplace_weight,
)


@wp.kernel
def Kernel_correct_eisph(P_sph: EISPHptl,
                         P_bnd: EISPHptl,
                         grid_sph: wp.uint64,
                         grid_bnd: wp.uint64,
                         support: float,
                         h: float,
                         dt: float) -> None:
    """
    Project the predictor velocity with the pressure difference gradient.

    P_sph, P_bnd: fluid and ghost EISPH particle structures
    grid_sph, grid_bnd: fixed-position HashGrid handles
    support: neighbour-query radius 2h [m]
    h: smoothing length [m]
    dt: time interval [s]

    Index convention: i / j are fluid subject / neighbour and bj is a ghost
    neighbour. The correction is u = u_star - dt*grad(p)/rho.

    # Output
    P_sph.vel[i], P_sph.acc[i]. P_sph.pos remains unchanged.
    """
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid_sph, tid)
    ri = P_sph.pos[i]
    pi = P_sph.pres[i]
    correction = P_sph.kgc[i]
    grad_p = wp.vec3(0.0, 0.0, 0.0)
    support2 = support * support

    # EISPH - EISPH pressure-difference gradient.
    for j in wp.hash_grid_query(grid_sph, ri, support):
        rij = ri - P_sph.pos[j]
        r2 = wp.dot(rij, rij)
        if r2 > R2_EPS and r2 < support2:
            grad_w = _correct_gradient(correction, _gradient(rij, r2, h))
            volume = P_sph.m[j] / P_sph.rho[j]
            grad_p = grad_p + volume * (P_sph.pres[j] - pi) * grad_w

    # EISPH - Bnd pressure-difference gradient.
    for bj in wp.hash_grid_query(grid_bnd, ri, support):
        ribj = ri - P_bnd.pos[bj]
        r2 = wp.dot(ribj, ribj)
        if r2 > R2_EPS and r2 < support2:
            grad_w = _correct_gradient(correction, _gradient(ribj, r2, h))
            volume = P_bnd.m[bj] / P_bnd.rho[bj]
            grad_p = grad_p + volume * (P_bnd.pres[bj] - pi) * grad_w

    old_velocity = P_sph.vel[i]
    velocity = P_sph.vel_star[i] - dt * grad_p / P_sph.rho[i]
    P_sph.vel[i] = velocity
    P_sph.acc[i] = (velocity - old_velocity) / dt


@wp.kernel
def Kernel_diagnostics_eisph(P_sph: EISPHptl,
                             P_bnd: EISPHptl,
                             grid_sph: wp.uint64,
                             grid_bnd: wp.uint64,
                             support: float,
                             h: float) -> None:
    """
    Calculate corrected divergence and the final PPE residual.

    P_sph, P_bnd: fluid and ghost EISPH particle structures
    grid_sph, grid_bnd: fixed-position HashGrid handles
    support: neighbour-query radius 2h [m]
    h: smoothing length [m]

    Index convention: i / j are fluid subject / neighbour and bj is a ghost
    neighbour. The residual is A*p-bi for the PDF Laplacian convention.

    # Output
    P_sph.divergence[i], P_sph.residual[i]
    """
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid_sph, tid)
    ri = P_sph.pos[i]
    vi = P_sph.vel[i]
    pi = P_sph.pres[i]
    correction = P_sph.kgc[i]
    div_u = float(0.0)
    operator_p = float(0.0)
    support2 = support * support

    # EISPH - EISPH corrected divergence and pressure operator.
    for j in wp.hash_grid_query(grid_sph, ri, support):
        rij = ri - P_sph.pos[j]
        r2 = wp.dot(rij, rij)
        if r2 > R2_EPS and r2 < support2:
            grad_w = _correct_gradient(correction, _gradient(rij, r2, h))
            volume = P_sph.m[j] / P_sph.rho[j]
            div_u = div_u + volume * wp.dot(P_sph.vel[j] - vi, grad_w)
            coefficient = _laplace_weight(volume, rij, grad_w, r2)
            operator_p = operator_p + coefficient * (pi - P_sph.pres[j])

    # EISPH - Bnd corrected divergence and Neumann pressure contribution.
    for bj in wp.hash_grid_query(grid_bnd, ri, support):
        ribj = ri - P_bnd.pos[bj]
        r2 = wp.dot(ribj, ribj)
        if r2 > R2_EPS and r2 < support2:
            grad_w = _correct_gradient(correction, _gradient(ribj, r2, h))
            volume = P_bnd.m[bj] / P_bnd.rho[bj]
            div_u = div_u + volume * wp.dot(P_bnd.vel[bj] - vi, grad_w)
            coefficient = _laplace_weight(volume, ribj, grad_w, r2)
            operator_p = operator_p + coefficient * (pi - P_bnd.pres[bj])

    P_sph.divergence[i] = div_u
    P_sph.residual[i] = operator_p - P_sph.bi[i]

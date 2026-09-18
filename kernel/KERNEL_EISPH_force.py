import warp as wp

from input.struct_eisph import EISPHptl
from kernel.KERNEL_EISPH_KNL import (
    R2_EPS,
    _correct_gradient,
    _gradient,
    _laplace_weight,
)


@wp.kernel
def Kernel_advec_vis_eisph(P_sph: EISPHptl,
                           P_bnd: EISPHptl,
                           grid_sph: wp.uint64,
                           grid_bnd: wp.uint64,
                           support: float,
                           h: float,
                           nu: float,
                           dt: float) -> None:
    """
    Calculate the non-pressure Eulerian velocity predictor.

    P_sph, P_bnd: fluid and ghost EISPH particle structures
    grid_sph, grid_bnd: fixed-position HashGrid handles
    support: neighbour-query radius 2h [m]
    h: smoothing length [m]
    nu: kinematic viscosity [m^2/s]
    dt: time interval [s]

    Index convention: i / j are fluid subject / neighbour and bj is a ghost
    neighbour. The predictor contains -(u dot grad)u + nu*laplacian(u).

    # Output
    P_sph.acc[i], P_sph.vel_star[i]. P_sph.pos is not updated.
    """
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid_sph, tid)
    ri = P_sph.pos[i]
    vi = P_sph.vel[i]
    correction = P_sph.kgc[i]
    convection = wp.vec3(0.0, 0.0, 0.0)
    laplace = wp.vec3(0.0, 0.0, 0.0)
    support2 = support * support

    # EISPH - EISPH convection and viscous diffusion.
    for j in wp.hash_grid_query(grid_sph, ri, support):
        rij = ri - P_sph.pos[j]
        r2 = wp.dot(rij, rij)
        if r2 > R2_EPS and r2 < support2:
            grad_w = _correct_gradient(correction, _gradient(rij, r2, h))
            volume = P_sph.m[j] / P_sph.rho[j]
            dv = P_sph.vel[j] - vi
            convection = convection + volume * wp.dot(vi, grad_w) * dv
            coefficient = _laplace_weight(volume, rij, grad_w, r2)
            laplace = laplace - coefficient * dv

    # EISPH - Bnd terms use the Dirichlet ghost velocity prepared beforehand.
    for bj in wp.hash_grid_query(grid_bnd, ri, support):
        ribj = ri - P_bnd.pos[bj]
        r2 = wp.dot(ribj, ribj)
        if r2 > R2_EPS and r2 < support2:
            grad_w = _correct_gradient(correction, _gradient(ribj, r2, h))
            volume = P_bnd.m[bj] / P_bnd.rho[bj]
            dv = P_bnd.vel[bj] - vi
            convection = convection + volume * wp.dot(vi, grad_w) * dv
            coefficient = _laplace_weight(volume, ribj, grad_w, r2)
            laplace = laplace - coefficient * dv

    acceleration = -convection + nu * laplace
    P_sph.acc[i] = acceleration
    P_sph.vel_star[i] = vi + dt * acceleration

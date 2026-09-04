import warp as wp

from kernel.KERNEL_KNL import Kernel_dw_Wendland, R2_MIN
from source.struct import SPHptl, BNDptl


@wp.kernel
def Kernel_force_sph(P_sph: SPHptl,
                     P_bnd: BNDptl,
                     grid_sph: wp.uint64,
                     grid_bnd: wp.uint64,
                     support: float,
                     h: float,
                     mu: float,
                     g: float) -> None:
    """
    Calculate acceleration of SPH particles from SPH, BND

    Pressure force (symmetric form) + viscous force (Morris) + gravity.
    All of them share one neighbor loop, because splitting the kernel
    would cost one more hash grid query per term.

    Index convention: i is the particle this thread computes for, j is a SPH
    (fluid) neighbour, d is a BND (dummy boundary) neighbour.

    # Output
    P_sph.acc[i]
    """
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid_sph, tid)
    ri = P_sph.pos[i]
    vi = P_sph.vel[i]
    rhoi = P_sph.rho[i]
    presi = P_sph.pres[i]
    acci = wp.vec3(0.0, 0.0, 0.0)
    # Sph - Sph
    for j in wp.hash_grid_query(grid_sph, ri, support):
        rij = ri - P_sph.pos[j]
        tdist = wp.dot(rij, rij)
        if tdist > R2_MIN:                          # Exclude itself
            if tdist < support * support:
                r = wp.sqrt(tdist)
                dwij = Kernel_dw_Wendland(r, h) * (rij / r)     # grad_i Wij
                rhoj = P_sph.rho[j]
                presj = P_sph.pres[j]
                # Pressure force
                acci = acci - P_sph.m[j] * (presi / (rhoi * rhoi)
                                            + presj / (rhoj * rhoj)) * dwij
                # Viscous force (Morris)
                visc = 2.0 * mu * wp.dot(rij, dwij) \
                    / (rhoi * rhoj * (tdist + 0.01 * h * h))
                acci = acci + P_sph.m[j] * visc * (vi - P_sph.vel[j])
    # Sph - Bnd
    for d in wp.hash_grid_query(grid_bnd, ri, support):
        rid = ri - P_bnd.pos[d]
        tdist = wp.dot(rid, rid)
        if tdist > R2_MIN:
            if tdist < support * support:
                r = wp.sqrt(tdist)
                dwid = Kernel_dw_Wendland(r, h) * (rid / r)     # grad_i Wid
                rhod = P_bnd.rho[d]
                presd = P_bnd.pres[d]
                # Pressure force
                acci = acci - P_bnd.m[d] * (presi / (rhoi * rhoi)
                                            + presd / (rhod * rhod)) * dwid
                # Viscous force (Morris), no-slip with the wall velocity
                visc = 2.0 * mu * wp.dot(rid, dwid) \
                    / (rhoi * rhod * (tdist + 0.01 * h * h))
                acci = acci + P_bnd.m[d] * visc * (vi - P_bnd.vel[d])
    # Gravity
    P_sph.acc[i] = acci + wp.vec3(0.0, -g, 0.0)

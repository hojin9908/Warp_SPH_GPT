import warp as wp

from kernel.KERNEL_KNL import Kernel_w_Wendland, R2_MIN
from source.struct import SPHptl, BNDptl

@wp.kernel
def Kernel_shepard_sph(P_sph: SPHptl,
                       P_bnd: BNDptl,
                       grid_sph: wp.uint64,
                       grid_bnd: wp.uint64,
                       support: float,
                       h: float) -> None:
    """
    Calculate Shepard filter for SPH from SPH, BND

    Index convention: i is the particle this thread computes for, j is a SPH
    (fluid) neighbour, d is a BND (dummy boundary) neighbour.

    # Output
    P_sph.flt[i]
    """
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid_sph, tid)
    ri = P_sph.pos[i]
    flt = float(0.0)
    # Sph - Sph
    for j in wp.hash_grid_query(grid_sph, ri, support):
        rij = ri - P_sph.pos[j]
        tdist = wp.dot(rij, rij)
        if tdist < support * support:
            wij = Kernel_w_Wendland(wp.sqrt(tdist), h)
            flt = flt + (P_sph.m[j] / P_sph.rho[j]) * wij
    # Sph - Bnd
    for d in wp.hash_grid_query(grid_bnd, ri, support):
        rid = ri - P_bnd.pos[d]
        tdist = wp.dot(rid, rid)
        if tdist < support * support:
            wid = Kernel_w_Wendland(wp.sqrt(tdist), h)
            flt = flt + (P_bnd.m[d] / P_bnd.rho[d]) * wid
    P_sph.flt[i] = flt




@wp.kernel
def Kernel_density_sph(P_sph: SPHptl,
                       P_bnd: BNDptl,
                       grid_sph: wp.uint64,
                       grid_bnd: wp.uint64,
                       support: float,
                       h: float) -> None:
    """
    Calculate density of SPH from SPH, BND

    The kernel sum runs over the same neighbours as Kernel_shepard_sph, so that
    the sum and the filter P_sph.flt it is divided by stay consistent.

    Index convention: i is the particle this thread computes for, j is a SPH
    (fluid) neighbour, d is a BND (dummy boundary) neighbour.

    # Output
    P_sph.rho[i]
    """
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid_sph, tid)
    ri = P_sph.pos[i]
    rhoi = float(0.0)
    # Sph - Sph
    for j in wp.hash_grid_query(grid_sph, ri, support):
        rij = ri - P_sph.pos[j]
        tdist = wp.dot(rij, rij)
        if tdist < support * support:
            wij = Kernel_w_Wendland(wp.sqrt(tdist), h)
            rhoi = rhoi + P_sph.m[j] * wij
    # Sph - Bnd
    for d in wp.hash_grid_query(grid_bnd, ri, support):
        rid = ri - P_bnd.pos[d]
        tdist = wp.dot(rid, rid)
        if tdist < support * support:
            wid = Kernel_w_Wendland(wp.sqrt(tdist), h)
            rhoi = rhoi + P_bnd.m[d] * wid
    P_sph.rho[i] = rhoi / P_sph.flt[i]




@wp.kernel
def Kernel_density_bnd(P_sph: SPHptl,
                       P_bnd: BNDptl,
                       grid_sph: wp.uint64,
                       grid_bnd: wp.uint64,
                       support: float,
                       h: float) -> None:
    """
    Calculate density of BND from SPH, BND

    Direct summation without the Shepard filter. Normalizing a wall particle
    would drag its density back to rho0 and erase the compression that the
    approaching fluid causes, which is exactly what makes the wall push back.

    Index convention: i is the particle this thread computes for and indexes
    P_bnd here, j is a SPH (fluid) neighbour, d is a BND neighbour.

    # Output
    P_bnd.rho[i]
    """
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid_bnd, tid)
    ri = P_bnd.pos[i]
    rhoi = float(0.0)
    # Bnd - Sph
    for j in wp.hash_grid_query(grid_sph, ri, support):
        rij = ri - P_sph.pos[j]
        tdist = wp.dot(rij, rij)
        if tdist < support * support:
            wij = Kernel_w_Wendland(wp.sqrt(tdist), h)
            rhoi = rhoi + P_sph.m[j] * wij
    # Bnd - Bnd
    for d in wp.hash_grid_query(grid_bnd, ri, support):
        rid = ri - P_bnd.pos[d]
        tdist = wp.dot(rid, rid)
        if tdist < support * support:
            wid = Kernel_w_Wendland(wp.sqrt(tdist), h)
            rhoi = rhoi + P_bnd.m[d] * wid
    P_bnd.rho[i] = rhoi





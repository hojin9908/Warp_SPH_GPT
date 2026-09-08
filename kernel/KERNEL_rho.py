import warp as wp

from kernel.KERNEL_KNL import Kernel_w_Wendland, R2_MIN
from input.struct import SPHptl, BNDptl

@wp.kernel
def Kernel_shepard_sph(P_sph: SPHptl,
                       P_bnd: BNDptl,
                       grid_sph: wp.uint64,
                       grid_bnd: wp.uint64,
                       support: float,
                       h: float) -> None:
    """
    Calculate Shepard filter for SPH from SPH, BND

    Index convention: i / j are the SPH (fluid) subject and neighbour,
    bi / bj are the BND (dummy boundary) subject and neighbour.

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
    for bj in wp.hash_grid_query(grid_bnd, ri, support):
        ribj = ri - P_bnd.pos[bj]
        tdist = wp.dot(ribj, ribj)
        if tdist < support * support:
            wibj = Kernel_w_Wendland(wp.sqrt(tdist), h)
            flt = flt + (P_bnd.m[bj] / P_bnd.rho[bj]) * wibj
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

    Index convention: i / j are the SPH (fluid) subject and neighbour,
    bi / bj are the BND (dummy boundary) subject and neighbour.

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
    for bj in wp.hash_grid_query(grid_bnd, ri, support):
        ribj = ri - P_bnd.pos[bj]
        tdist = wp.dot(ribj, ribj)
        if tdist < support * support:
            wibj = Kernel_w_Wendland(wp.sqrt(tdist), h)
            rhoi = rhoi + P_bnd.m[bj] * wibj
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

    Index convention: i / j are the SPH (fluid) subject and neighbour,
    bi / bj are the BND (dummy boundary) subject and neighbour. The subject
    of this kernel is a boundary particle, so it is bi.

    # Output
    P_bnd.rho[bi]
    """
    tid = wp.tid()
    bi = wp.hash_grid_point_id(grid_bnd, tid)
    rbi = P_bnd.pos[bi]
    rhobi = float(0.0)
    # Bnd - Sph
    for j in wp.hash_grid_query(grid_sph, rbi, support):
        rbij = rbi - P_sph.pos[j]
        tdist = wp.dot(rbij, rbij)
        if tdist < support * support:
            wbij = Kernel_w_Wendland(wp.sqrt(tdist), h)
            rhobi = rhobi + P_sph.m[j] * wbij
    # Bnd - Bnd
    for bj in wp.hash_grid_query(grid_bnd, rbi, support):
        rbibj = rbi - P_bnd.pos[bj]
        tdist = wp.dot(rbibj, rbibj)
        if tdist < support * support:
            wbibj = Kernel_w_Wendland(wp.sqrt(tdist), h)
            rhobi = rhobi + P_bnd.m[bj] * wbibj
    P_bnd.rho[bi] = rhobi





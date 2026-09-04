import warp as wp

from source.struct import SPHptl, BNDptl


@wp.kernel
def Kernel_pres_sph(P_sph: SPHptl,
                    rho0: float,
                    c0: float,
                    gamma: float) -> None:
    """
    Calculate pressure of SPH particles with Tait EOS

    pres = B * ((rho/rho0)^gamma - 1),   B = rho0 * c0^2 / gamma
    Negative pressure is clamped to zero to avoid tensile instability.

    # Output
    P_sph.pres[i]
    """
    i = wp.tid()
    B = rho0 * c0 * c0 / gamma
    q = wp.pow(P_sph.rho[i] / rho0, gamma) - 1.0
    P_sph.pres[i] = B * wp.max(q, 0.0)


@wp.kernel
def Kernel_pres_bnd(P_bnd: BNDptl,
                    rho0: float,
                    c0: float,
                    gamma: float) -> None:
    """
    Calculate pressure of BND particles with Tait EOS

    Same EOS as Kernel_pres_sph. Boundary particles need their own pressure
    because Kernel_force_sph reads P_bnd.pres[bj] for the repulsive wall force.

    The subject of this kernel is a boundary particle, so it is bi.

    # Output
    P_bnd.pres[bi]
    """
    bi = wp.tid()
    B = rho0 * c0 * c0 / gamma
    q = wp.pow(P_bnd.rho[bi] / rho0, gamma) - 1.0
    P_bnd.pres[bi] = B * wp.max(q, 0.0)

import warp as wp

from input.struct_eisph import EISPHptl


@wp.kernel
def Kernel_Dirichlet_BC(P_sph: EISPHptl,
                        P_bnd: EISPHptl,
                        predicted: int) -> None:
    """
    Apply a prescribed velocity Dirichlet condition through mirrored ghosts.

    P_sph: fixed fluid EISPH particles
    P_bnd: fixed ghost particles with vel_bc and mirror fields
    predicted: 0 updates vel, nonzero updates vel_star

    The subject is a ghost particle bi and j=P_bnd.mirror[bi] is its stable
    fluid index. P_bnd.vel_bc[bi] is the arbitrary prescribed velocity
    u_D(x_boundary,t), which may differ for every boundary point and step.
    The mirrored-ghost relation is u_bi = 2*u_D - u_j.

    # Output
    P_bnd.vel[bi] or P_bnd.vel_star[bi]
    """
    bi = wp.tid()
    j = P_bnd.mirror[bi]
    if predicted == 0:
        P_bnd.vel[bi] = 2.0 * P_bnd.vel_bc[bi] - P_sph.vel[j]
    else:
        P_bnd.vel_star[bi] = 2.0 * P_bnd.vel_bc[bi] - P_sph.vel_star[j]


@wp.kernel
def Kernel_pressure_boundary_eisph(P_sph: EISPHptl,
                                   P_bnd: EISPHptl) -> None:
    """
    Apply zero-normal-gradient pressure to every ghost point.

    The subject is boundary particle bi and j=P_bnd.mirror[bi] is the stable
    mirrored fluid index.

    # Output
    P_bnd.pres[bi]
    """
    bi = wp.tid()
    j = P_bnd.mirror[bi]
    P_bnd.pres[bi] = P_sph.pres[j]

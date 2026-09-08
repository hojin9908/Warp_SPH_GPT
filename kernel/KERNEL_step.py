import warp as wp

from input.struct import SPHptl


@wp.kernel
def Kernel_step_sph(P_sph: SPHptl,
                    dt: float) -> None:
    """
    Update velocity and position of SPH particles (semi-implicit Euler)

    vel and pos are updated in one kernel because pos must be advanced
    with the already updated vel. BND particles never move, so they are
    not touched here.

    # Output
    P_sph.vel[i], P_sph.pos[i]
    """
    i = wp.tid()
    vel = P_sph.vel[i] + P_sph.acc[i] * dt
    P_sph.vel[i] = vel
    P_sph.pos[i] = P_sph.pos[i] + vel * dt

import warp as wp

from input.struct import DEMptl


@wp.kernel
def Kernel_step_dem(P_dem: DEMptl, dt: float) -> None:
    """
    Update moving DEM velocity, position and spin with semi-implicit Euler.

    All contact, gravity and SPH coupling forces must be accumulated first.
    Position advances with the updated velocity, just as in Kernel_step_sph.

    P_dem: Particle structure of moving DEM spheres [N_dem]
    dt: time interval [s]
    Index convention: a is the DEM subject; no neighbour query is needed.

    # Output
    P_dem.acc[a], P_dem.vel[a], P_dem.pos[a], P_dem.omega[a]
    Fixed DEM boundaries are excluded from this kernel.
    """
    a = wp.tid()
    # Translation: convert force to acceleration and update velocity first.
    acc = P_dem.force[a] / P_dem.m[a]
    vel = P_dem.vel[a] + acc * dt
    P_dem.acc[a] = acc
    P_dem.vel[a] = vel
    P_dem.pos[a] = P_dem.pos[a] + vel * dt
    # Rotation: a solid sphere has the same scalar moment of inertia about every axis.
    P_dem.omega[a] = P_dem.omega[a] + P_dem.torque[a] / P_dem.inertia[a] * dt

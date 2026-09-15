import warp as wp

from input.struct import DEMptl

# The DEM path is forward-only; no backward state is retained.
wp.set_module_options({"enable_backward": False})


@wp.kernel
def Kernel_count_dem_contacts(P_dem: DEMptl,
                              grid_dem: wp.uint64,
                              radius_max: float) -> None:
    """Write each DEM row's active contact count to *_offset_new."""
    a = wp.tid()
    ra = P_dem.pos[a]
    count = int(0)
    for b in wp.hash_grid_query(grid_dem, ra, P_dem.radius[a] + radius_max):
        if b != a:
            distance = wp.length(P_dem.pos[b] - ra)
            if P_dem.radius[a] + P_dem.radius[b] - distance > 0.0:
                count = count + 1
    P_dem.contact_dem_offset_new[a] = count


@wp.func
def DEM_contact(normal: wp.vec3, overlap: float,
                radius_a: float, radius_b: float,
                vel_a: wp.vec3, vel_b: wp.vec3,
                omega_a: wp.vec3, omega_b: wp.vec3,
                tangent: wp.vec3, K: float, eta: float, mu: float, dt: float):
    """
    Calculate one DEM contact using a linear spring/dashpot and history friction.

    normal: unit vector from subject a to neighbour b [-]
    overlap: positive overlap Ra + Rb - distance [m]
    radius_a, radius_b: subject / neighbour sphere radii [m]
    vel_a, vel_b: translational velocities at the sphere centers [m/s]
    omega_a, omega_b: angular velocities about all three axes [rad/s]
    tangent: tangential displacement stored for this contact at the previous step [m]
    K: normal / tangential spring stiffness [N/m]
    eta: normal / tangential damping coefficient [N s/m]
    mu: Coulomb friction coefficient [-]
    dt: time interval used to update the contact history [s]

    # Output
    return: force on a [N], updated displacement [m], torque on a [N m]
    Each moving particle thread computes its own side of a contact, so the
    neighbour force and torque do not need to be returned or stored.

    Reference: function_DEM_INTERACTION.cuh. Its componentwise sliding limiter
    and unprojected displacement history are retained deliberately.
    """
    # Contact lever arms include half the overlap on each sphere.
    rc_a = (radius_a - 0.5 * overlap) * normal
    rc_b = -(radius_b - 0.5 * overlap) * normal
    # Relative contact-point velocity includes translation and rotation.
    urel = vel_b + wp.cross(omega_b, rc_b) - vel_a - wp.cross(omega_a, rc_a)
    un = wp.dot(urel, normal) * normal             # normal relative velocity [m/s]
    us = urel - un                                # tangential relative velocity [m/s]
    # Integrate tangential slip and regularize its direction at zero slip speed.
    displacement = tangent + us * dt
    direction = us / (wp.length(us) + 1.0e-20)
    # Normal force: repulsive spring plus relative-velocity damping.
    fn = -K * overlap * normal + eta * un
    # Tangential force: history spring / damping, limited by Coulomb friction.
    fss = K * displacement + eta * us
    fsf = mu * K * overlap * direction
    fs = fss
    if wp.length(fss) >= wp.length(fsf):
        fs = wp.vec3(wp.sign(fss[0]) * wp.abs(fsf[0]),
                     wp.sign(fss[1]) * wp.abs(fsf[1]),
                     wp.sign(fss[2]) * wp.abs(fsf[2]))
    # Tangential force gives the subject torque; the normal force is center-directed.
    return fn + fs, displacement, wp.cross(rc_a, fs)


@wp.kernel
def Kernel_force_dem(P_dem: DEMptl,
                     grid_dem: wp.uint64,
                     radius_max: float,
                     g: float,
                     dt: float) -> None:
    """
    Calculate DEM loads while rebuilding one compact output CSR.

    P_dem owns both CSR sides. This kernel reads the *_old fields and writes
    the scanned/allocated *_new fields. The caller swaps the fields afterward.

    P_dem: moving DEM particles plus old/read and new/write contact CSR fields
    grid_dem: HashGrid handle built from current DEM positions
    radius_max: largest moving DEM neighbour radius [m]
    g: gravity magnitude in the negative y direction [m/s^2]
    dt: interval used to integrate tangential displacement [s]
    # Output
    P_dem.force[a], P_dem.torque[a]
    P_dem.contact_dem_id_new/tang_dem_new contain every active directed pair.
    """
    a = wp.tid()
    ra = P_dem.pos[a]
    force = wp.vec3(0.0, -P_dem.m[a] * g, 0.0)
    torque = wp.vec3(0.0, 0.0, 0.0)
    # First writable edge of particle a in this step's scanned new CSR.
    output_index = P_dem.contact_dem_offset_new[a]

    for b in wp.hash_grid_query(grid_dem, ra, P_dem.radius[a] + radius_max):
        if b != a:
            rab = P_dem.pos[b] - ra
            dist = wp.length(rab)
            overlap = P_dem.radius[a] + P_dem.radius[b] - dist
            if overlap > 0.0:
                tangent = wp.vec3(0.0, 0.0, 0.0)
                # Stable ID b selects the matching tangential history from
                # the previous/current (*_old) CSR row of particle a.
                for old_index in range(P_dem.contact_dem_offset_old[a],
                                       P_dem.contact_dem_offset_old[a + 1]):
                    if P_dem.contact_dem_id_old[old_index] == b:
                        tangent = P_dem.tang_dem_old[old_index]

                normal = wp.vec3(1.0, 0.0, 0.0)
                if a > b:
                    normal = -normal
                if dist > 1.0e-12:
                    normal = rab / dist
                fab, displacement, torque_a = DEM_contact(
                    normal, overlap, P_dem.radius[a], P_dem.radius[b],
                    P_dem.vel[a], P_dem.vel[b], P_dem.omega[a], P_dem.omega[b],
                    tangent, P_dem.K[a], P_dem.eta[a], P_dem.mu[a], dt)
                force = force + fab
                torque = torque + torque_a
                P_dem.contact_dem_id_new[output_index] = b
                P_dem.tang_dem_new[output_index] = displacement
                output_index = output_index + 1

    P_dem.force[a] = force
    P_dem.torque[a] = torque

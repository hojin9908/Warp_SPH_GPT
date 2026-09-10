import warp as wp

from input.struct import DEMptl, DEMBNDptl
from kernel.KERNEL_DEM_force import DEM_contact

# The DEM path is forward-only; no backward state is retained.
wp.set_module_options({"enable_backward": False})


@wp.kernel
def Kernel_reset_dem_bnd(P_dem_bnd: DEMBNDptl,
                         grid_dem_bnd: wp.uint64,
                         radius_max: float,
                         g: float) -> None:
    """Reset fixed-boundary loads using fixed-fixed normal contacts and gravity."""
    dbi = wp.tid()
    rdbi = P_dem_bnd.pos[dbi]
    force = wp.vec3(0.0, -P_dem_bnd.m[dbi] * g, 0.0)

    for dbj in wp.hash_grid_query(
            grid_dem_bnd, rdbi, P_dem_bnd.radius[dbi] + radius_max):
        if dbi != dbj:
            rdb = P_dem_bnd.pos[dbj] - rdbi
            dist = wp.length(rdb)
            overlap = P_dem_bnd.radius[dbi] + P_dem_bnd.radius[dbj] - dist
            if dist > 1.0e-12 and overlap > 0.0:
                force = force - P_dem_bnd.K[dbi] * overlap * rdb / dist

    P_dem_bnd.force[dbi] = force
    P_dem_bnd.torque[dbi] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def Kernel_count_bnd_contacts(P_dem: DEMptl,
                              P_dem_bnd: DEMBNDptl,
                              grid_dem_bnd: wp.uint64,
                              radius_max: float) -> None:
    """Write each DEM row's active boundary count to *_offset_new."""
    a = wp.tid()
    ra = P_dem.pos[a]
    count = int(0)
    for dbj in wp.hash_grid_query(
            grid_dem_bnd, ra, P_dem.radius[a] + radius_max):
        distance = wp.length(P_dem_bnd.pos[dbj] - ra)
        if P_dem.radius[a] + P_dem_bnd.radius[dbj] - distance > 0.0:
            count = count + 1
    P_dem.contact_bnd_offset_new[a] = count


@wp.kernel
def Kernel_bc_dem(P_dem: DEMptl,
                  P_dem_bnd: DEMBNDptl,
                  grid_dem_bnd: wp.uint64,
                  radius_max: float,
                  dt: float) -> None:
    """
    Add moving-fixed loads while rebuilding one compact output CSR.

    P_dem owns both boundary CSR sides. This kernel reads the *_old fields and
    writes the scanned/allocated *_new fields. The caller swaps them afterward.

    P_dem: moving DEM particles plus old/read and new/write boundary CSR fields
    P_dem_bnd: fixed boundary spheres
    grid_dem_bnd: HashGrid handle for fixed boundary positions
    radius_max: largest boundary sphere radius [m]
    dt: interval used to integrate tangential displacement [s]
    # Output
    Wall force/torque is added to P_dem and atomically reacted on P_dem_bnd.
    P_dem.contact_bnd_id_new/tang_bnd_new contain every active pair.
    """
    a = wp.tid()
    ra = P_dem.pos[a]
    force = wp.vec3(0.0, 0.0, 0.0)
    torque = wp.vec3(0.0, 0.0, 0.0)
    # First writable boundary edge of particle a in this step's new CSR.
    output_index = P_dem.contact_bnd_offset_new[a]

    for dbj in wp.hash_grid_query(
            grid_dem_bnd, ra, P_dem.radius[a] + radius_max):
        radb = P_dem_bnd.pos[dbj] - ra
        dist = wp.length(radb)
        overlap = P_dem.radius[a] + P_dem_bnd.radius[dbj] - dist
        if overlap > 0.0:
            tangent = wp.vec3(0.0, 0.0, 0.0)
            # Match stable boundary ID dbj against the old/read CSR history.
            for old_index in range(P_dem.contact_bnd_offset_old[a],
                                   P_dem.contact_bnd_offset_old[a + 1]):
                if P_dem.contact_bnd_id_old[old_index] == dbj:
                    tangent = P_dem.tang_bnd_old[old_index]

            normal = wp.vec3(0.0, -1.0, 0.0)
            if dist > 1.0e-12:
                normal = radb / dist
            fadb, displacement, torque_a, torque_dbj = DEM_contact(
                normal, overlap, P_dem.radius[a], P_dem_bnd.radius[dbj],
                P_dem.vel[a], P_dem_bnd.vel[dbj],
                P_dem.omega[a], P_dem_bnd.omega[dbj],
                tangent, P_dem.K[a], P_dem.eta[a], P_dem.mu[a], dt)
            force = force + fadb
            torque = torque + torque_a
            P_dem.contact_bnd_id_new[output_index] = dbj
            P_dem.tang_bnd_new[output_index] = displacement
            output_index = output_index + 1
            # Several moving particles can contact the same fixed sphere.
            wp.atomic_add(P_dem_bnd.force, dbj, -fadb)
            wp.atomic_add(P_dem_bnd.torque, dbj, torque_dbj)

    P_dem.force[a] = P_dem.force[a] + force
    P_dem.torque[a] = P_dem.torque[a] + torque


@wp.kernel
def Kernel_acc_dem_bnd(P_dem_bnd: DEMBNDptl) -> None:
    """Store force/mass for diagnostics without integrating the fixed wall."""
    dbi = wp.tid()
    P_dem_bnd.acc[dbi] = P_dem_bnd.force[dbi] / P_dem_bnd.m[dbi]

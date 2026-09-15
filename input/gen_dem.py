import numpy as np
import warp as wp

from input.Config import Solv
from input.struct import DEMptl, DEMBNDptl


class DEMPtlGeneration:
    """
    Generate DEM and fixed DEM boundary structures for the 3D dam-break case.

    Moving spheres form an nx-by-ny-by-nz block; fixed spheres cover five walls.
    Material, initial motion and geometry are read from the supplied Solv object.
    """

    def __init__(self, solv: Solv) -> None:
        """Keep the simulation settings used to generate positions and SoA arrays."""
        self.solv = solv

    def dem_particle(self) -> np.ndarray:
        """
        Generate moving DEM sphere centers on a regular xyz lattice.

        The origin specifies the first sphere center, not the edge of the block.
        The top is open; only the bottom and side walls constrain the centers.

        return: numpy positions [N_dem, 3] [m], with y vertical and z depth
        """
        solv = self.solv
        # Moving DEM centers, ordered independently of later HashGrid sorting.
        gx, gy, gz = np.meshgrid(
            solv.dem_origin_x + np.arange(solv.dem_nx) * solv.dem_spacing,
            solv.dem_origin_y + np.arange(solv.dem_ny) * solv.dem_spacing,
            solv.dem_origin_z + np.arange(solv.dem_nz) * solv.dem_spacing,
            indexing="ij")
        pos = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
        # Check the sphere extent against all five closed sides of the tank.
        if (pos[:, 0].min() < solv.dem_radius or
                pos[:, 0].max() + solv.dem_radius > solv.tank_width or
                pos[:, 1].min() < solv.dem_radius or
                pos[:, 2].min() < solv.dem_radius or
                pos[:, 2].max() + solv.dem_radius > solv.tank_depth):
            raise ValueError("initial DEM spheres must lie inside the tank walls")
        return pos

    def boundary_particle(self) -> np.ndarray:
        """
        Generate fixed DEM spheres on the bottom and four vertical tank walls.

        Wall centers lie one boundary radius outside the fluid tank. A shell
        mask assigns shared edges / corners once, with no ceiling at y=height.

        return: numpy positions [N_dem_bnd, 3] [m]
        """
        solv = self.solv
        radius = solv.dem_bnd_radius
        # Equal spacing no larger than the configured spacing, including corners.
        nx = int(np.ceil((solv.tank_width + 2.0 * radius) / solv.dem_bnd_spacing))
        ny = int(np.ceil((solv.tank_height + radius) / solv.dem_bnd_spacing))
        nz = int(np.ceil((solv.tank_depth + 2.0 * radius) / solv.dem_bnd_spacing))
        x = np.linspace(-radius, solv.tank_width + radius, nx + 1)
        y = np.linspace(-radius, solv.tank_height, ny + 1)
        z = np.linspace(-radius, solv.tank_depth + radius, nz + 1)
        gx, gy, gz = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1),
                                np.arange(nz + 1), indexing="ij")
        wall = ((gy == 0) | (gx == 0) | (gx == nx) | (gz == 0) | (gz == nz))
        return np.stack([x[gx[wall]], y[gy[wall]], z[gz[wall]]], axis=1)

    def build(self) -> tuple[DEMptl, DEMBNDptl]:
        """
        Allocate moving DEM and fixed DEM boundary SoA arrays.

        return: dem (DEMptl), bnd (DEMBNDptl)
        All arrays are allocated on solv.device. Dynamic fields and contact
        history belong to moving DEM particles; the fixed wall stores no loads.
        """
        solv = self.solv
        solv.validate_dem()
        dev = solv.device
        dem_pos = self.dem_particle()
        bnd_pos = self.boundary_particle()
        n_dem, n_bnd = len(dem_pos), len(bnd_pos)
        dem = DEMptl()
        bnd = DEMBNDptl()
        # Shared geometry/material metadata; only moving spheres own dynamic loads.
        for P, pos, radius, rho, mass, volume, inertia in (
                (dem, dem_pos, solv.dem_radius, solv.dem_rho, solv.dem_mass,
                 solv.dem_volume, solv.dem_inertia),
                (bnd, bnd_pos, solv.dem_bnd_radius, solv.dem_bnd_rho, solv.dem_bnd_mass,
                 solv.dem_bnd_volume, solv.dem_bnd_inertia)):
            n = len(pos)
            P.pos = wp.array(pos, dtype=wp.vec3, device=dev)
            for key in ("vel", "omega"):
                setattr(P, key, wp.zeros(n, dtype=wp.vec3, device=dev))
            for key, value in (("radius", radius), ("rho", rho), ("m", mass),
                               ("volume", volume), ("inertia", inertia),
                               ("K", solv.dem_K), ("eta", solv.dem_eta), ("mu", solv.dem_mu)):
                setattr(P, key, wp.full(n, value, dtype=float, device=dev))
        # Acceleration, force and torque are integrated only for moving spheres.
        for key in ("acc", "force", "torque"):
            setattr(dem, key, wp.zeros(n_dem, dtype=wp.vec3, device=dev))
        # Initial translation / spin uses all three axes; boundary motion stays zero.
        dem.vel.fill_(wp.vec3(solv.dem_vel_x, solv.dem_vel_y, solv.dem_vel_z))
        dem.omega.fill_(wp.vec3(solv.dem_omega_x, solv.dem_omega_y, solv.dem_omega_z))
        # Both CSR sides start with E=0. During a step *_old is the history
        # input and *_new is the output; the two sides are swapped afterward.
        dem.contact_dem_offset_old = wp.zeros(n_dem + 1, dtype=wp.int32, device=dev)
        dem.contact_dem_id_old = wp.empty(0, dtype=wp.int32, device=dev)
        dem.tang_dem_old = wp.empty(0, dtype=wp.vec3, device=dev)
        dem.contact_dem_offset_new = wp.zeros(n_dem + 1, dtype=wp.int32, device=dev)
        dem.contact_dem_id_new = wp.empty(0, dtype=wp.int32, device=dev)
        dem.tang_dem_new = wp.empty(0, dtype=wp.vec3, device=dev)
        dem.contact_bnd_offset_old = wp.zeros(n_dem + 1, dtype=wp.int32, device=dev)
        dem.contact_bnd_id_old = wp.empty(0, dtype=wp.int32, device=dev)
        dem.tang_bnd_old = wp.empty(0, dtype=wp.vec3, device=dev)
        dem.contact_bnd_offset_new = wp.zeros(n_dem + 1, dtype=wp.int32, device=dev)
        dem.contact_bnd_id_new = wp.empty(0, dtype=wp.int32, device=dev)
        dem.tang_bnd_new = wp.empty(0, dtype=wp.vec3, device=dev)
        # Fluid interpolation and force buffers at moving DEM positions.
        dem.flt_s = wp.zeros(n_dem, dtype=float, device=dev)
        dem.porosity = wp.full(n_dem, 1.0, dtype=float, device=dev)
        dem.drag = wp.zeros(n_dem, dtype=wp.vec3, device=dev)
        dem.pressure_force = wp.zeros(n_dem, dtype=wp.vec3, device=dev)
        return dem, bnd

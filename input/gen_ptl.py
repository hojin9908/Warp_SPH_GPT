import numpy as np
import warp as wp

from input.struct import SPHptl, BNDptl
from input.Config_SPH_DEM import Solv

class DamPtlGeneration:
    """
    Generate particle structures for a 3D dam break (z vertical, y depth).
    """

    def __init__(self, solv: Solv) -> None:
        """Keep the shared configuration and SPH lattice spacing."""
        self.solv = solv
        self.dx = solv.dx
        self.bnd_layer = solv.bnd_layer

    def fluid_particle(self) -> np.ndarray:
        """Returns fluid particle positions as numpy array [N_sph, 3]."""
        solv = self.solv
        dx = self.dx
        n_x = int(round(solv.fluid_width / dx))
        n_y = int(round(solv.fluid_depth / dx))
        n_z = int(round(solv.fluid_height / dx))
        grid_x, grid_y, grid_z = np.meshgrid(
            solv.fluid_origin_x + np.arange(n_x) * dx,
            solv.fluid_origin_y + np.arange(n_y) * dx,
            solv.fluid_origin_z + np.arange(n_z) * dx,
            indexing="ij"
        )
        x = grid_x.ravel()
        y = grid_y.ravel()
        z = grid_z.ravel()
        return np.stack([x, y, z], axis=1)  # [N_sph, 3]

    def boundary_particle(self) -> np.ndarray:
        """Returns boundary particle positions as numpy array [N_bnd, 3].

        Five dummy walls surround the tank: bottom, left, right, front and back.
        Integer lattice indices assign each edge / corner exactly once.
        Side walls extend bnd layers above the opening; no ceiling is generated.
        """
        solv = self.solv
        dx = self.dx
        bnd = self.bnd_layer

        n_x_tank = int(round(solv.tank_width / dx))
        n_y_tank = int(round(solv.tank_depth / dx))
        n_z_tank = int(round(solv.tank_height / dx))
        # A single shell mask prevents duplicated corner masses in the density sum.
        gx, gy, gz = np.meshgrid(np.arange(-bnd, n_x_tank + bnd),
                                np.arange(-bnd, n_y_tank + bnd),
                                np.arange(-bnd, n_z_tank + bnd), indexing="ij")
        wall = ((gz < 0) | (gx < 0) | (gx >= n_x_tank)
                | (gy < 0) | (gy >= n_y_tank))
        return np.stack([gx[wall], gy[wall], gz[wall]], axis=1) * dx

    def build(self) -> tuple[SPHptl, BNDptl]:
        """Build SPHptl and BNDptl structs from the dam-break initial configuration.

        Returns:
            sph: SPHptl  -- fluid particles  (p_type=1)
            bnd: BNDptl  -- boundary particles (p_type=0)
        """
        solv = self.solv

        solv.validate_sph()

        fluid_pos = self.fluid_particle()    # [N_sph, 3]
        bnd_pos   = self.boundary_particle() # [N_bnd, 3]

        n_sph = len(fluid_pos)
        n_bnd = len(bnd_pos)

        dev = solv.device
        mass = solv.rho0 * solv.dx ** 3        # 3D particle mass [kg]

        sph = SPHptl()
        sph.pos     = wp.array(fluid_pos, dtype=wp.vec3, device=dev)
        sph.vel     = wp.zeros(n_sph, dtype=wp.vec3,  device=dev)
        sph.rho_raw = wp.full(n_sph, solv.rho0, dtype=float, device=dev)
        sph.rho     = wp.full(n_sph, solv.rho0, dtype=float, device=dev)
        sph.pres    = wp.zeros(n_sph, dtype=float,     device=dev)
        sph.acc     = wp.zeros(n_sph, dtype=wp.vec3,  device=dev)
        sph.m       = wp.full(n_sph, mass, dtype=float, device=dev)
        sph.flt     = wp.zeros(n_sph, dtype=float, device=dev)
        # SPH-DEM fields at fluid positions; SPH-only runs keep these initial values.
        sph.porosity = wp.full(n_sph, 1.0, dtype=float, device=dev)
        sph.pgf      = wp.zeros(n_sph, dtype=wp.vec3, device=dev)
        sph.acc_dem  = wp.zeros(n_sph, dtype=wp.vec3, device=dev)

        bnd = BNDptl()
        bnd.pos     = wp.array(bnd_pos, dtype=wp.vec3, device=dev)
        bnd.vel     = wp.zeros(n_bnd, dtype=wp.vec3,  device=dev)
        bnd.rho_raw = wp.full(n_bnd, solv.rho0, dtype=float, device=dev)
        bnd.rho     = wp.full(n_bnd, solv.rho0, dtype=float, device=dev)
        bnd.pres    = wp.zeros(n_bnd, dtype=float,     device=dev)
        bnd.acc     = wp.zeros(n_bnd, dtype=wp.vec3,  device=dev)
        bnd.m       = wp.full(n_bnd, mass, dtype=float, device=dev)

        return sph, bnd

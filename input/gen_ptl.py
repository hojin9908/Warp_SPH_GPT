import numpy as np
import warp as wp

from input.struct import SPHptl, BNDptl
from input.Config import Solv

class DamPtlGeneration:
    """
    Generate Particle Structure for 2D Dam Break Simulation
    """

    def __init__(self, solv: Solv) -> None:
        self.solv = solv
        self.dx = solv.dx
        self.bnd_layer = solv.bnd_layer

    def fluid_particle(self) -> np.ndarray:
        """Returns fluid particle positions as numpy array [N_sph, 3]."""
        solv = self.solv
        dx = self.dx
        n_x = int(round(solv.fluid_width / dx))
        n_y = int(round(solv.fluid_height / dx))
        grid_x, grid_y = np.meshgrid(
            solv.fluid_origin_x + np.arange(n_x) * dx,
            solv.fluid_origin_y + np.arange(n_y) * dx,
            indexing="ij"
        )
        x = grid_x.ravel()
        y = grid_y.ravel()
        return np.stack([x, y, np.zeros_like(x)], axis=1)  # [N_sph, 3]

    def boundary_particle(self) -> np.ndarray:
        """Returns boundary particle positions as numpy array [N_bnd, 3].

        Tank geometry (open top):
          - Bottom wall: y in [-bnd*dx, -dx], x covers full width + corners
          - Left  wall:  x in [-bnd*dx, -dx], y in [0, tank_height + bnd*dx)
          - Right wall:  x in [tank_width, tank_width+(bnd-1)*dx], same y range
        """
        solv = self.solv
        dx = self.dx
        bnd = self.bnd_layer

        n_x_tank = int(round(solv.tank_width / dx))
        n_y_wall  = int(round(solv.tank_height / dx)) + bnd

        # Bottom wall including corner regions
        x_btm = np.arange(-bnd, n_x_tank + bnd) * dx
        y_btm = np.arange(-bnd, 0) * dx
        gx, gy = np.meshgrid(x_btm, y_btm, indexing="ij")
        btm = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)

        # Left wall (y >= 0, corners already covered by bottom)
        x_lft  = np.arange(-bnd, 0) * dx
        y_wall = np.arange(0, n_y_wall) * dx
        gx, gy = np.meshgrid(x_lft, y_wall, indexing="ij")
        lft = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)

        # Right wall (y >= 0, corners already covered by bottom)
        x_rgt = solv.tank_width + np.arange(0, bnd) * dx
        gx, gy = np.meshgrid(x_rgt, y_wall, indexing="ij")
        rgt = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)

        return np.vstack([btm, lft, rgt])  # [N_bnd, 3]

    def build(self) -> tuple[SPHptl, BNDptl]:
        """Build SPHptl and BNDptl structs from the dam-break initial configuration.

        Returns:
            sph: SPHptl  -- fluid particles  (p_type=1)
            bnd: BNDptl  -- boundary particles (p_type=0)
        """
        solv = self.solv

        fluid_pos = self.fluid_particle()    # [N_sph, 3]
        bnd_pos   = self.boundary_particle() # [N_bnd, 3]

        n_sph = len(fluid_pos)
        n_bnd = len(bnd_pos)

        dev = solv.device
        mass = solv.rho0 * solv.dx * solv.dx    # 2D mass

        sph = SPHptl()
        sph.pos     = wp.array(fluid_pos, dtype=wp.vec3, device=dev)
        sph.vel     = wp.zeros(n_sph, dtype=wp.vec3,  device=dev)
        sph.rho_raw = wp.full(n_sph, solv.rho0, dtype=float, device=dev)
        sph.rho     = wp.full(n_sph, solv.rho0, dtype=float, device=dev)
        sph.pres    = wp.zeros(n_sph, dtype=float,     device=dev)
        sph.acc     = wp.zeros(n_sph, dtype=wp.vec3,  device=dev)
        sph.m       = wp.full(n_sph, mass, dtype=float, device=dev)
        sph.flt     = wp.zeros(n_sph, dtype=float, device=dev)

        bnd = BNDptl()
        bnd.pos     = wp.array(bnd_pos, dtype=wp.vec3, device=dev)
        bnd.vel     = wp.zeros(n_bnd, dtype=wp.vec3,  device=dev)
        bnd.rho_raw = wp.full(n_bnd, solv.rho0, dtype=float, device=dev)
        bnd.rho     = wp.full(n_bnd, solv.rho0, dtype=float, device=dev)
        bnd.pres    = wp.zeros(n_bnd, dtype=float,     device=dev)
        bnd.acc     = wp.zeros(n_bnd, dtype=wp.vec3,  device=dev)
        bnd.m       = wp.full(n_bnd, mass, dtype=float, device=dev)

        return sph, bnd

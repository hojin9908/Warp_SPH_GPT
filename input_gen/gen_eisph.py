import numpy as np
import warp as wp

from input_gen.config import GenerationConfig as Solv
from input.struct_eisph import EISPHptl


class CavityPtlGeneration:
    """
    Generate fixed EISPH particle structures for a lid-driven cavity.

    The square cavity lies in the x-z plane. Fluid points are cell-centred,
    while ghost points continue the same lattice outside the four walls.
    """

    def __init__(self, solv: Solv) -> None:
        """Keep the EISPH configuration."""
        self.solv = solv

    def fluid_particle(self) -> np.ndarray:
        """Return cell-centred fluid positions as a numpy array [N_sph, 3]."""
        n = self.solv.cells
        axis = (np.arange(n, dtype=np.float32) + 0.5) * self.solv.dx
        grid_x, grid_z = np.meshgrid(axis, axis, indexing="xy")
        return np.stack((grid_x.ravel(), np.zeros(grid_x.size, np.float32),
                         grid_z.ravel()), axis=1)

    def boundary_particle(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return ghost positions, Dirichlet velocities and mirrored fluid indices.

        return: position [N_bnd,3], prescribed velocity [N_bnd,3],
            and stable mirrored fluid index [N_bnd].
        """
        n = self.solv.cells
        layers = self.solv.bnd_layer
        ids = np.arange(-layers, n + layers, dtype=np.int32)
        grid_x, grid_z = np.meshgrid(ids, ids, indexing="xy")
        # One shell mask creates all four walls without interior lattice points.
        shell = ((grid_x < 0) | (grid_x >= n) |
                 (grid_z < 0) | (grid_z >= n))
        ix = grid_x[shell]
        iz = grid_z[shell]
        pos = np.stack(((ix + 0.5) * self.solv.dx, np.zeros(ix.size),
                        (iz + 0.5) * self.solv.dx), axis=1).astype(np.float32)

        vel_bc = np.zeros_like(pos)
        # The cavity is one prescribed-velocity case: a moving lid and fixed walls.
        lid = (iz >= n) & (ix >= 0) & (ix < n)
        vel_bc[lid, 0] = self.solv.lid_velocity

        # Reflect every ghost lattice coordinate through its closest wall.
        mirror_x = np.where(ix < 0, -ix - 1,
                            np.where(ix >= n, 2 * n - ix - 1, ix))
        mirror_z = np.where(iz < 0, -iz - 1,
                            np.where(iz >= n, 2 * n - iz - 1, iz))
        mirror = (mirror_z * n + mirror_x).astype(np.int32)
        if np.any((mirror < 0) | (mirror >= n * n)):
            raise ValueError("bnd_layer cannot exceed the cavity cell count")
        return pos, vel_bc, mirror

    def _build_particle(self,
                        pos: np.ndarray,
                        vel: np.ndarray,
                        vel_bc: np.ndarray,
                        mirror: np.ndarray) -> EISPHptl:
        """
        Allocate one EISPHptl structure on solv.device.

        pos, vel, vel_bc: host arrays with shape [N_eisph,3]
        mirror: stable fluid indices used by ghost particles [N_eisph]

        return: initialized EISPH particle structure
        """
        solv = self.solv
        n = len(pos)
        mass = solv.rho0 * solv.dx ** 2  # 2D particle mass per unit depth [kg/m]
        particle = EISPHptl()
        particle.pos = wp.array(pos, dtype=wp.vec3, device=solv.device)
        particle.vel = wp.array(vel, dtype=wp.vec3, device=solv.device)
        particle.vel_star = wp.array(vel, dtype=wp.vec3, device=solv.device)
        particle.vel_bc = wp.array(vel_bc, dtype=wp.vec3, device=solv.device)
        particle.acc = wp.zeros(n, dtype=wp.vec3, device=solv.device)
        particle.kgc = wp.zeros(n, dtype=wp.mat22, device=solv.device)
        particle.rho = wp.full(n, solv.rho0, dtype=float, device=solv.device)
        particle.m = wp.full(n, mass, dtype=float, device=solv.device)
        particle.pres = wp.zeros(n, dtype=float, device=solv.device)
        particle.Aij = wp.zeros(n, dtype=float, device=solv.device)
        particle.bi = wp.zeros(n, dtype=float, device=solv.device)
        particle.Aijpj = wp.zeros(n, dtype=float, device=solv.device)
        particle.divergence = wp.zeros(n, dtype=float, device=solv.device)
        particle.residual = wp.zeros(n, dtype=float, device=solv.device)
        particle.mirror = wp.array(mirror, dtype=wp.int32, device=solv.device)
        return particle

    def build(self) -> tuple[EISPHptl, EISPHptl]:
        """
        Build fluid and ghost particles from the cavity configuration.

        # Output
        P_sph: EISPHptl for fixed fluid points [N_sph]
        P_bnd: EISPHptl for fixed ghost boundary points [N_bnd]
        """
        self.solv.validate_eisph()
        fluid_pos = self.fluid_particle()  # [N_sph,3]
        boundary_pos, vel_bc, boundary_mirror = self.boundary_particle()
        fluid_vel = np.zeros_like(fluid_pos)
        fluid_ids = np.arange(len(fluid_pos), dtype=np.int32)
        zero_vel_bc = np.zeros_like(fluid_pos)
        # Initial fluid velocity is zero, so u_b = 2*u_D - u_mirror.
        boundary_vel = 2.0 * vel_bc
        return (self._build_particle(fluid_pos, fluid_vel, zero_vel_bc, fluid_ids),
                self._build_particle(boundary_pos, boundary_vel,
                                     vel_bc, boundary_mirror))

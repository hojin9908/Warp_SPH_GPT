import numpy as np

from input.Config_SPH_DEM import Solv
from input.struct import SPHptl, BNDptl, DEMptl, DEMBNDptl
from output.gif_gen import collect_frame, save_gif
from output.output import save_vtk, save_dem_vtk, save_pvd


class FrameWriter:
    """
    Collect and write synchronized SPH/DEM simulation output.

    solv: simulation and output configuration
    P_sph, P_bnd: fluid and fixed SPH particle structures
    P_dem, P_dem_bnd: optional moving and fixed DEM particle structures
    """

    def __init__(self,
                 solv: Solv,
                 P_sph: SPHptl,
                 P_bnd: BNDptl,
                 P_dem: DEMptl = None,
                 P_dem_bnd: DEMBNDptl = None):
        if (P_dem is None) != (P_dem_bnd is None):
            raise ValueError("provide both moving and fixed DEM particles")

        self.solv = solv
        self.P_sph = P_sph
        self.P_bnd = P_bnd
        self.P_dem = P_dem
        self.P_dem_bnd = P_dem_bnd
        self.coupled = P_dem is not None

        # VTP references are retained until their PVD collections are written.
        self.frames: list[tuple[str, float]] = []
        self.dem_frames: list[tuple[str, float]] = []
        # Device state is copied to host only when a GIF was requested.
        self.gif_frames: list[dict] = []

    def write_frame(self, step: int) -> None:
        """
        Write one state after `step` integrations; step 0 is the initial state.

        Raise FloatingPointError before writing if a principal particle field
        contains NaN or infinity.
        """
        t = step * self.solv.dt
        particle_fields = [
            (self.P_sph, ("pos", "vel", "acc", "rho")),
            (self.P_bnd, ("pos", "vel", "acc", "rho")),
        ]
        if self.coupled:
            particle_fields.extend([
                (self.P_dem, ("pos", "vel", "acc", "rho")),
                (self.P_dem_bnd, ("pos", "vel", "rho")),
            ])

        # Reject invalid states before writing any part of this synchronized frame.
        for particles_phase, fields in particle_fields:
            for key in fields:
                if not np.isfinite(getattr(particles_phase, key).numpy()).all():
                    raise FloatingPointError(f"non-finite {key} at step {step}")

        self.frames.append(save_vtk(self.P_sph, self.P_bnd, step, t,
                                    out_dir=self.solv.output_dir, name="sph"))
        if self.coupled:
            self.dem_frames.append(save_dem_vtk(
                self.P_dem, self.P_dem_bnd, step, t,
                out_dir=self.solv.output_dir))
        if self.solv.gif_save:
            self.gif_frames.append(collect_frame(
                self.P_sph, self.P_bnd, t, self.P_dem, self.P_dem_bnd))

    def finalize(self) -> None:
        """Write the GIF and phase-specific PVD collections after simulation."""
        if self.gif_frames:
            gif_name = "dam_break_sph_dem_3d" if self.coupled else "dam_break_3d"
            gif = save_gif(self.gif_frames, self.solv,
                           ani_dir=self.solv.animation_dir, name=gif_name)
            print(f"\n[output] {len(self.gif_frames)} frames -> {gif}")

        if self.frames:
            path = save_pvd(self.frames, out_dir=self.solv.output_dir, name="sph")
            print(f"\n[output] {len(self.frames)} frames -> {path}\n")
        if self.dem_frames:
            path = save_pvd(self.dem_frames, out_dir=self.solv.output_dir, name="dem")
            print(f"[output] {len(self.dem_frames)} frames -> {path}\n")

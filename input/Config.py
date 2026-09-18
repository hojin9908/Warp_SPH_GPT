"""Solver controls, analogous to SOPHIA solv.txt.

Geometry and initial particle properties live in input_file/*.txt. The example
preprocessor settings live in input_gen/config.py. h is the common SPH/EISPH
kernel length; DEM coupling h is stored on each DEM particle.
"""
from dataclasses import dataclass
import math


@dataclass
class Solv:
    # Input files are resolved relative to this project, or use an absolute path.
    input_dir: str = "input_file"
    device: str = "cuda:0"

    # Hydrodynamic solver / material (EISPH example defaults).
    h: float = 0.052
    rho0: float = 1.0
    c0: float = 31.3209
    gamma: float = 7.0
    mu: float = 0.05
    g: float = 9.81
    nu: float = 0.01  # EISPH kinematic viscosity [m^2/s], formerly U*L/Re
    shepard_step: int = 20
    grid_slice: int = 64

    # Fixed time stepping.
    dt: float = 2.0e-3
    n_steps: int = 5000

    # Coupling controls; K, eta, mu, h belong to DEM particle arrays.
    dem_enable: bool = False
    dem_porosity_min: float = 0.05
    dem_porosity_max: float = 1.0
    dem_dt_safety: float = 0.1

    # Output controls.
    output_step: int = 100
    gif_save: bool = True
    gif_fps: int = 20
    output_dir: str = "result/3d"
    animation_dir: str = "animation/3d"
    animation_path: str = "animation/lid_driven_cavity.gif"

    @property
    def support(self) -> float:
        """The implemented Wendland kernels have support 2*h."""
        return 2.0 * self.h

    def _validate_common(self) -> None:
        for name in ("h", "rho0", "dt"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("n_steps", "output_step", "shepard_step"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("grid_slice", "gif_fps"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    def validate_sph(self) -> None:
        """Validate solver settings; input_reader validates particle data."""
        self._validate_common()
        for name in ("c0", "gamma"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("mu", "g"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.dt > 0.25 * self.h / self.c0:
            raise ValueError("dt exceeds the initial SPH acoustic limit 0.25*h/c0")
        self.validate_dem()

    def validate_dem(self) -> None:
        """Validate coupling controls, independently of any generated lattice."""
        if not self.dem_enable:
            return
        if not 0.0 < self.dem_porosity_min <= self.dem_porosity_max <= 1.0:
            raise ValueError("DEM porosity bounds must satisfy 0 < min <= max <= 1")
        if not math.isfinite(self.dem_dt_safety) or self.dem_dt_safety <= 0.0:
            raise ValueError("dem_dt_safety must be finite and positive")

    def validate_eisph(self) -> None:
        """Validate EISPH controls; loaded points supply geometry and wall speeds."""
        self._validate_common()
        if not math.isfinite(self.nu) or self.nu < 0.0:
            raise ValueError("nu must be finite and nonnegative")

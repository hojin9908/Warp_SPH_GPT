from dataclasses import dataclass

from input.Config import Solv as _Solv


@dataclass
class Solv(_Solv):
    """Use the shared solver configuration with the SPH-DEM defaults."""

    dx: float = 0.02
    rho0: float = 1000.0
    dt: float = 1.0e-4
    n_steps: int = 9000
    output_step: int = 90
    dem_enable: bool = True

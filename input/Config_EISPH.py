from dataclasses import dataclass
import math


@dataclass
class EISPHConfig:
    """
    # project property
    device: 계산에 쓸 디바이스 ("cuda:0", "cpu" ...)

    # simulation setting
    length: 정사각 cavity의 x/z 방향 길이 [m]
    dx: 고정 Eulerian 입자 간격 [m]
    h_factor: smoothing length 배수 (h = h_factor * dx)
    boundary_layers: ghost boundary particle 겹 수

    # physical coefficient
    rho0: 기준 밀도 [kg/m^3]
    lid_velocity: 상부 lid의 +x 방향 속도 [m/s]
    reynolds: Reynolds number, Re = U_lid * L / nu [-]

    # time integration
    dt: 시간 간격 [s]
    n_steps: 순방향 시뮬레이션 스텝 수

    # output
    output_step: GIF용 속도장을 host로 복사하는 주기 [step]
    gif_fps: GIF 재생 속도 [frame/s]
    animation_path: lid-driven cavity GIF 저장 경로

    # hash grid
    grid_slice: HashGrid 해시 버킷 한 변의 개수

    # derived property
    h: smoothing length [m]
    support: Wendland kernel support radius 2h [m]
    nu: 동점성계수 U_lid * L / Re [m^2/s]
    cells: cavity 한 변의 유체 입자 수
    """

    # project property
    device: str = "cuda:0"

    # simulation setting
    length: float = 1.0
    dx: float = 0.04
    h_factor: float = 1.3
    boundary_layers: int = 3

    # physical coefficient
    rho0: float = 1.0
    lid_velocity: float = 1.0
    reynolds: float = 100.0

    # time integration
    dt: float = 2.0e-3
    n_steps: int = 5000

    # output
    output_step: int = 100
    gif_fps: int = 20
    animation_path: str = "animation/lid_driven_cavity.gif"

    # hash grid
    grid_slice: int = 64

    @property
    def h(self) -> float:
        """Return the Wendland smoothing length h = h_factor * dx [m]."""
        return self.h_factor * self.dx

    @property
    def support(self) -> float:
        """Return the compact kernel support radius 2h [m]."""
        return 2.0 * self.h

    @property
    def nu(self) -> float:
        """Return kinematic viscosity from the requested Reynolds number [m^2/s]."""
        return self.lid_velocity * self.length / self.reynolds

    @property
    def cells(self) -> int:
        """Return the number of cell-centred fluid points along one side."""
        return int(round(self.length / self.dx))

    def validate(self) -> None:
        """
        Check the EISPH cavity lattice, material and fixed time step.

        The advection and viscosity bounds are conservative checks for the
        explicit predictor. The time interval is not changed automatically.

        # Output
        Raise ValueError for an invalid setting; all configuration values stay unchanged.
        """
        for name in ("length", "dx", "h_factor", "rho0", "lid_velocity",
                     "reynolds", "dt"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("boundary_layers", "n_steps", "output_step",
                     "gif_fps", "grid_slice"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.cells < 5 or not math.isclose(self.cells * self.dx, self.length,
                                               rel_tol=0.0, abs_tol=1.0e-10):
            raise ValueError("length must be an integer multiple of dx with at least 5 cells")
        if self.boundary_layers < math.ceil(self.support / self.dx):
            raise ValueError("boundary_layers must cover the complete kernel support")
        if self.boundary_layers > self.cells:
            raise ValueError("boundary_layers cannot exceed the cavity cell count")
        if self.dt > 0.25 * self.dx / self.lid_velocity:
            raise ValueError("dt exceeds the explicit advection limit")
        if self.dt > 0.125 * self.dx * self.dx / self.nu:
            raise ValueError("dt exceeds the explicit viscosity limit")

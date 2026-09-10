from dataclasses import dataclass
from typing import Any
import math

@dataclass
class Solv:
    """
    # project property
    device: 계산에 쓸 디바이스 ("cuda:0", "cpu" ...)

    # simulation setting
    dx: 입자 간격
    tank_width: 수조 내부 폭
    tank_height: 수조 내부 높이 (위쪽은 열려 있다)
    tank_depth: 수조 내부 z 방향 깊이 [m]
    fluid_width: 초기 유체 블록 폭
    fluid_height: 초기 유체 블록 높이
    fluid_depth: 초기 유체 블록 깊이 [m]
    fluid_origin_x: 유체 블록 왼쪽 아래 모서리 x
    fluid_origin_y: 유체 블록 왼쪽 아래 모서리 y
    fluid_origin_z: 초기 유체 블록의 앞쪽 모서리 z [m]
    bnd_layer: dummy boundary particle 겹 수
    h: smoothing length
    support: Support radius

    # physical coefficient
    rho0: 기준 밀도
    gamma: Tait 지수 (1<=gamma<=7)
    c0: 수치 음속 [m/s]
    mu: 점성계수 [Pa s]
    g: 중력 가속도
    h_factor: 기준 smoothing length 배수 (기본 h = h_factor * dx)

    # density filter
    shepard_step: Shepard filter 적용 주기 [step]

    # PDE solver hyperparameter
    dt: 시간 간격 [s]. SPH 음향 조건과 DEM 접촉 시간 척도 이하로 설정
    n_steps: 순방향 시뮬레이션 스텝 수

    # output
    output_step: 입자 상태를 vtk 로 뽑는 주기 [step]. 0 이면 출력하지 않는다
    gif_save: True 면 같은 주기의 프레임으로 gif 애니메이션도 만든다
    output_dir: SPH/DEM VTP와 PVD 파일을 저장할 폴더
    animation_dir: SPH/DEM 통합 GIF를 저장할 폴더

    # hash grid
    grid_slice: HashGrid 해시 버킷 한 변의 개수

    # DEM setting (3D 구)
    dem_enable: True 면 SPH-DEM 연계, False 면 기존 SPH 계산만 수행
    dem_radius: 이동 DEM 구 반지름 [m]
    dem_rho: DEM 재료 밀도 [kg/m^3]
    dem_K: 정상/접선 스프링 강성 [N/m]
    dem_eta: 정상/접선 감쇠계수 [N s/m]
    dem_mu: Coulomb 마찰계수 [-]
    dem_inertia_factor: 관성모멘트 계수. I = factor * m * radius^2 [-]
    dem_nx, dem_ny, dem_nz: 초기 DEM 배치의 x/y/z 방향 입자 수
    dem_spacing: 초기 DEM 중심 간격 [m]
    dem_origin_x, dem_origin_y, dem_origin_z: 초기 DEM 배열 첫 입자의 중심 좌표 [m]
    dem_vel_x, dem_vel_y, dem_vel_z: 초기 DEM 병진속도 [m/s]
    dem_omega_x, dem_omega_y, dem_omega_z: 초기 DEM 각속도 [rad/s]

    # DEM boundary setting
    dem_bnd_radius: 고정 DEM 경계 구 반지름 [m]
    dem_bnd_rho: 고정 DEM 경계 재료 밀도 [kg/m^3]
    dem_bnd_spacing: 벽면 격자의 경계 구 중심 간격 상한 [m]

    # SPH-DEM coupling
    dem_h: 보간과 항력 반작용이 공유하는 smoothing length [m]
    dem_porosity_min, dem_porosity_max: 항력에 사용하는 공극률의 하한/상한 [-]
    dem_dt_safety: 접촉 시간 척도에 곱하는 시간 간격 안전계수 [-]

    # derived DEM property
    dem_volume, dem_bnd_volume: 구 체적 (4/3) * pi * radius^3 [m^3]
    dem_mass, dem_bnd_mass: 질량 rho * volume [kg]
    dem_inertia, dem_bnd_inertia: 구의 각 축에 대한 관성모멘트 [kg m^2]
    dem_support: 연계 커널 지지 반경 2 * dem_h [m]
    """
    # project property
    device: str = "cuda:0"

    # simulation setting
    dx: float = 0.02
    tank_width: float = 2.0
    tank_height: float = 1.0
    tank_depth: float = 0.4
    fluid_width: float = 0.5
    fluid_height: float = 0.5
    fluid_depth: float = 0.4
    fluid_origin_x: float = 0.0
    fluid_origin_y: float = 0.0
    fluid_origin_z: float = 0.0
    bnd_layer: int = 3
    h: float = 1.3 * 0.02
    support: float = 2.0 * (1.3 * 0.02)

    # physical coefficient
    rho0: float = 1000.0
    gamma: float = 7.0
    c0: float = 31.3209
    mu: float = 0.05
    g: float = 9.81
    h_factor: float = 1.3

    # density filter
    shepard_step: int = 20

    # PDE solver hyperparameter
    dt: float = 1.0e-4
    n_steps: int = 9000

    # output
    output_step: int = 90
    gif_save: bool = True
    output_dir: str = "result/3d"
    animation_dir: str = "animation/3d"

    # hash grid
    grid_slice: int = 64

    # DEM setting (3D solid spheres; SI mass, force and torque)
    dem_enable: bool = True
    dem_radius: float = 0.025
    dem_rho: float = 2500.0
    dem_K: float = 2.0e4                  # normal / tangential spring stiffness [N/m]
    dem_eta: float = 4.0                  # normal / tangential damping [N s/m]
    dem_mu: float = 0.3                   # Coulomb friction coefficient
    dem_inertia_factor: float = 0.4       # I = (2/5) * m * radius^2 (solid sphere)
    dem_nx: int = 5
    dem_ny: int = 2
    dem_nz: int = 2
    dem_spacing: float = 0.065
    dem_origin_x: float = 0.12
    dem_origin_y: float = 0.62
    dem_origin_z: float = 0.1675
    dem_vel_x: float = 0.0
    dem_vel_y: float = 0.0
    dem_vel_z: float = 0.0
    dem_omega_x: float = 0.0              # initial angular velocity about x [rad/s]
    dem_omega_y: float = 0.0              # initial angular velocity about y [rad/s]
    dem_omega_z: float = 0.0              # initial angular velocity about z [rad/s]

    # Fixed DEM boundary spheres (same contact law, no integration)
    dem_bnd_radius: float = 0.01
    dem_bnd_rho: float = 2500.0
    dem_bnd_spacing: float = 0.02

    # SPH-DEM coupling: common support for interpolation and back reaction
    dem_h: float = 0.08
    dem_porosity_min: float = 0.05
    dem_porosity_max: float = 1.0
    dem_dt_safety: float = 0.1

    @property
    def dem_volume(self) -> float:
        """Return moving DEM sphere volume (4/3) * pi * R^3 [m^3]."""
        return (4.0 / 3.0) * math.pi * self.dem_radius ** 3

    @property
    def dem_mass(self) -> float:
        """Return moving DEM sphere mass rho * volume [kg]."""
        return self.dem_rho * self.dem_volume

    @property
    def dem_inertia(self) -> float:
        """Return spherical DEM moment of inertia about each axis [kg m^2]."""
        return self.dem_inertia_factor * self.dem_mass * self.dem_radius ** 2

    @property
    def dem_bnd_volume(self) -> float:
        """Return fixed DEM boundary sphere volume (4/3) * pi * R_bnd^3 [m^3]."""
        return (4.0 / 3.0) * math.pi * self.dem_bnd_radius ** 3

    @property
    def dem_bnd_mass(self) -> float:
        """Return fixed DEM boundary sphere mass rho_bnd * volume [kg]."""
        return self.dem_bnd_rho * self.dem_bnd_volume

    @property
    def dem_bnd_inertia(self) -> float:
        """Return fixed DEM boundary moment of inertia about each axis [kg m^2]."""
        return self.dem_inertia_factor * self.dem_bnd_mass * self.dem_bnd_radius ** 2

    @property
    def dem_support(self) -> float:
        """Return the common Wendland support radius for SPH-DEM exchange [m]."""
        return 2.0 * self.dem_h

    def validate_sph(self) -> None:
        """
        Check the 3D SPH domain, lattice, material and fixed time step.

        # Output
        Raise ValueError for invalid settings; all configuration values stay unchanged.
        The acoustic bound is a conservative initial-state check, not adaptive stepping.
        """
        for name in ("dx", "h", "support", "tank_width", "tank_height", "tank_depth",
                     "fluid_width", "fluid_height", "fluid_depth", "rho0", "c0", "gamma", "dt"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isclose(self.support, 2.0 * self.h):
            raise ValueError("Wendland support must equal 2*h")
        if self.dt > 0.25 * self.h / self.c0:
            raise ValueError("dt exceeds the initial SPH acoustic limit 0.25*h/c0")
        for name in ("g", "mu"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("bnd_layer", "grid_slice"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("n_steps", "output_step", "shepard_step"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        # The integer wall lattice uses tank lengths that are multiples of dx.
        for name in ("tank_width", "tank_height", "tank_depth"):
            cells = getattr(self, name) / self.dx
            if not math.isclose(cells, round(cells), abs_tol=1.0e-8):
                raise ValueError(f"{name} must be an integer multiple of dx")
        for axis, extent, tank in (("x", self.fluid_width, self.tank_width),
                                   ("y", self.fluid_height, self.tank_height),
                                   ("z", self.fluid_depth, self.tank_depth)):
            origin = getattr(self, "fluid_origin_" + axis)
            if (not math.isfinite(origin) or origin < 0.0 or origin + extent > tank + 1.0e-12
                    or round(extent / self.dx) < 1):
                raise ValueError(f"initial fluid block must fit the tank along {axis}")

    def validate_dem(self) -> None:
        """
        Check DEM material, initial spacing, porosity bounds and contact time step.

        All values are read from this Solv instance; no setting is changed here.
        The time-step bound uses the reduced mass of two identical moving spheres.

        # Output
        Raise ValueError for an invalid checked setting; return None otherwise.
        DEM-disabled runs skip these checks.
        """
        if not self.dem_enable:
            return
        # Material / length scales that enter denominators must be positive.
        for name in ("dem_radius", "dem_rho", "dem_K", "dem_inertia_factor",
                     "dem_bnd_radius", "dem_bnd_rho", "dem_bnd_spacing",
                     "dem_h", "dem_dt_safety", "dt"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        # Zero damping or friction is allowed for frictionless / elastic cases.
        if (not math.isfinite(self.dem_eta) or not math.isfinite(self.dem_mu)
                or self.dem_eta < 0.0 or self.dem_mu < 0.0):
            raise ValueError("dem_eta and dem_mu must be nonnegative")
        # Moving spheres must not overlap initially, along any lattice direction.
        if any(not isinstance(n, int) or n < 1 for n in (self.dem_nx, self.dem_ny, self.dem_nz)):
            raise ValueError("dem_nx, dem_ny and dem_nz must be positive integers")
        if not math.isfinite(self.dem_spacing) or self.dem_spacing < 2.0 * self.dem_radius:
            raise ValueError("initial DEM particles must not overlap")
        if self.dem_bnd_spacing > 2.0 * self.dem_bnd_radius:
            raise ValueError("DEM boundary spacing must not exceed the boundary diameter")
        # A moving center cannot pass through a square pore of the wall lattice.
        if self.dem_bnd_spacing / math.sqrt(2.0) >= self.dem_radius + self.dem_bnd_radius:
            raise ValueError("DEM boundary pores must be smaller than the moving spheres")
        for name in ("dem_origin_x", "dem_origin_y", "dem_origin_z", "dem_vel_x",
                     "dem_vel_y", "dem_vel_z", "dem_omega_x", "dem_omega_y", "dem_omega_z"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        # Reaction acceleration divides by the clamped fluid fraction.
        if not 0.0 < self.dem_porosity_min <= self.dem_porosity_max <= 1.0:
            raise ValueError("DEM porosity bounds must satisfy 0 < min <= max <= 1")
        # Elastic time scale sqrt(m_eff / K) and damping time scale m_eff / eta.
        # A fixed wall has infinite inertia; two moving spheres give the smaller m_eff.
        mass_eff = 0.5 * self.dem_mass
        dt_contact = self.dem_dt_safety * min(
            math.sqrt(mass_eff / self.dem_K),
            mass_eff / max(self.dem_eta, 1.0e-20))
        if self.dt > dt_contact:
            raise ValueError(f"dt exceeds DEM contact limit {dt_contact:.6g} s")

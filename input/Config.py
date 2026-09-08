from dataclasses import dataclass
from typing import Any

@dataclass
class Solv:
    """
    # project property
    device: 계산에 쓸 디바이스 ("cuda:0", "cpu" ...)

    # simulation setting
    dx: 입자 간격
    tank_width: 수조 내부 폭
    tank_height: 수조 내부 높이 (위쪽은 열려 있다)
    fluid_width: 초기 유체 블록 폭
    fluid_height: 초기 유체 블록 높이
    fluid_origin_x: 유체 블록 왼쪽 아래 모서리 x
    fluid_origin_y: 유체 블록 왼쪽 아래 모서리 y
    bnd_layer: dummy boundary particle 겹 수
    h: smoothing length
    support: Support radius

    # physical coefficient
    rho0: 기준 밀도
    gamma: Tait 지수 (1<=gamma<=7)
    c0: 수치 음속. 0 이면 10*sqrt(2*g*fluid_height) 로 자동 계산
    mu: 점성계수 [Pa s]
    g: 중력 가속도
    h: smoothing length. 0 이면 h_factor * dx 로 자동 계산
    h_factor: h 자동 계산에 쓰는 배수 (h = h_factor * dx)

    # density filter
    shepard_step: Shepard filter 적용 주기 [step]

    # PDE solver hyperparameter
    dt: 시간 간격. 0 이면 cfl * h / c0 로 자동 계산
    n_steps: 순방향 시뮬레이션 스텝 수

    # output
    output_step: 입자 상태를 vtk 로 뽑는 주기 [step]. 0 이면 출력하지 않는다
    gif_save: True 면 같은 주기의 프레임으로 gif 애니메이션도 만든다

    # hash grid
    grid_slice: HashGrid 해시 버킷 한 변의 개수
    """
    # project property
    device: str = "cuda:0"

    # simulation setting
    dx: float = 0.02
    tank_width: float = 2.0
    tank_height: float = 1.0
    fluid_width: float = 0.5
    fluid_height: float = 0.5
    fluid_origin_x: float = 0.0
    fluid_origin_y: float = 0.0
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
    dt: float = 2.0753e-4
    n_steps: int = 4000

    # output
    output_step: int = 40
    gif_save: bool = True

    # hash grid
    grid_slice: int = 128
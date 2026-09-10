import warp as wp
import math

# To Exclude itself
R2_MIN = wp.constant(1.0e-12)

@wp.func
def Kernel_w_Wendland(r: float, h: float) -> float:
    """
    Calculate the 3D Wendland C2 kernel value [1/m^3].

    r: Distance between two particles [m]
    h: Smoothing length [m]

    wij: Wendland kernel value
    """
    q = r / h
    wij = float(0.0)
    u = 1.0 - 0.5 * q
    if q < 2.0:
        # The 3D normalization satisfies integral W dV = 1 over a sphere of radius 2h.
        wij = 21.0 * u * u * u * u * (1.0 + 2.0 * q) / (16.0 * math.pi * h * h * h)
    return wij


@wp.func
def Kernel_dw_Wendland(r: float, h: float) -> float:
    """
    Calculate dw/dr of the 3D Wendland C2 kernel [1/m^4].

    r: Distance between two particles
    h: Smoothing length

    dwij: dw/dr of Wendland kernel value
    """
    q = r / h
    dwij = float(0.0)
    if q < 2.0:
        u = 1.0 - 0.5 * q
        dwij = 21.0 * (-5.0 * q * u * u * u) / (16.0 * math.pi * h * h * h * h)
    return dwij


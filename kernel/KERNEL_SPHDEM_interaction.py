import warp as wp

from input.struct import SPHptl, BNDptl, DEMptl
from kernel.KERNEL_KNL import Kernel_w_Wendland, Kernel_dw_Wendland, R2_MIN

# Forward simulation only. Avoid generating oversized adjoint argument lists
# for the coupled SoA structs on GPUs with a 4 KiB kernel-parameter limit.
wp.set_module_options({"enable_backward": False})


@wp.func
def DEM_drag_beta(eps: float, rho: float, mu: float, diameter: float, speed: float) -> float:
    """
    Calculate beta / (1 - eps) with the reference Ergun / Wen-Yu drag law.

    eps: fluid fraction, already clamped by the calling kernel [-]
    rho: fluid density interpolated at the DEM position [kg/m^3]
    mu: dynamic fluid viscosity [Pa s]
    diameter: DEM diameter, 2 * radius [m]
    speed: magnitude of fluid velocity minus DEM velocity [m/s]

    # Output
    return: beta / (1 - eps) [kg/(m^3 s)]
    The solid fraction is cancelled analytically to avoid 0/0 as eps approaches 1.
    Reference: function_SPH_DEM_COUPLING.cuh.
    """
    beta = float(0.0)
    reynolds = rho * diameter * speed / (mu + 1.0e-20)  # particle Reynolds number [-]
    if eps <= 0.8:
        # Ergun: viscous resistance plus inertial resistance in a dense solid region.
        beta = 150.0 * (1.0 - eps) / eps * mu / (diameter * diameter) \
            + 1.75 * rho * speed / diameter
    else:
        # Wen-Yu: single-particle drag coefficient with a porosity correction.
        # Algebraic Cd*speed limit avoids 0/0 for stationary particles.
        cd_speed = 0.44 * speed
        if reynolds <= 1000.0:
            cd_speed = 24.0 * mu / (rho * diameter) * (1.0 + 0.15 * wp.pow(reynolds, 0.687))
        beta = 0.75 * cd_speed * rho / diameter * wp.pow(eps, -1.65)
    return beta


@wp.kernel
def Kernel_prep_sphdem(P_sph: SPHptl, P_bnd: BNDptl, P_dem: DEMptl,
                       grid_sph: wp.uint64, grid_bnd: wp.uint64,
                       grid_dem: wp.uint64, support: float, h: float,
                       sph_support: float, sph_h: float,
                       eps_min: float, eps_max: float) -> None:
    """
    Calculate fluid porosity and -grad(p) at SPH positions for SPH-DEM exchange.

    Index convention: i / j are the SPH subject / neighbour, bj is a dummy
    boundary neighbour and b is a moving DEM neighbour.

    P_sph: Particle structure of SPH particles [N_sph]
    P_bnd: Particle structure of SPH dummy boundary particles [N_bnd]
    P_dem: Particle structure of moving DEM spheres [N_dem]
    grid_sph, grid_bnd, grid_dem: HashGrid handles built from current positions
    support: maximum DEM coupling support radius [m]
    h: common DEM h when uniform, otherwise 0 (per-neighbour normalization)
    sph_support, sph_h: original SPH support radius and smoothing length [m]
    eps_min, eps_max: allowed fluid-fraction bounds [-]

    # Output
    P_sph.porosity[i] [-], P_sph.pgf[i] [Pa/m]
    Porosity uses fluid-only normalization; the pressure gradient includes walls.

    Reference: function_PREP.cuh and function_ALE.cuh (difference pressure
    gradient). Use this solver's uncorrected 3D Wendland gradient.
    """
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid_sph, tid)
    ri = P_sph.pos[i]
    flt = float(0.0)                              # fluid volume-weighted kernel sum [-]
    solid = float(0.0)                            # DEM volume-weighted kernel sum [-]
    pgf = wp.vec3(0.0, 0.0, 0.0)                 # negative pressure gradient [Pa/m]
    # Uniform h keeps the original inexpensive Shepard normalization exactly.
    if h > 0.0:
        for j in wp.hash_grid_query(grid_sph, ri, support):
            dist = wp.length(ri - P_sph.pos[j])
            if dist < support:
                flt = flt + P_sph.m[j] / P_sph.rho[j] * Kernel_w_Wendland(dist, h)
    # A heterogeneous cloud normalizes each solid contribution using that
    # neighbour's h and a fluid-only kernel sum centred at the SPH point.
    for b in wp.hash_grid_query(grid_dem, ri, support):
        hb = P_dem.h[b]
        dist = wp.length(ri - P_dem.pos[b])
        if dist < 2.0 * hb:
            local_flt = flt
            if h <= 0.0:
                local_flt = float(0.0)
                for j in wp.hash_grid_query(grid_sph, ri, 2.0 * hb):
                    fluid_distance = wp.length(ri - P_sph.pos[j])
                    if fluid_distance < 2.0 * hb:
                        local_flt = local_flt + P_sph.m[j] / P_sph.rho[j] * Kernel_w_Wendland(fluid_distance, hb)
            solid = solid + P_dem.volume[b] * Kernel_w_Wendland(dist, hb) / (local_flt + 1.0e-20)
    # Sph - Sph: difference pressure gradient using the original SPH support.
    for j in wp.hash_grid_query(grid_sph, ri, sph_support):
        rij = ri - P_sph.pos[j]
        r2 = wp.dot(rij, rij)
        if r2 > R2_MIN and r2 < sph_support * sph_support:
            dist = wp.sqrt(r2)
            dwij = Kernel_dw_Wendland(dist, sph_h) * rij / dist  # grad_i Wij
            pgf = pgf - P_sph.m[j] / P_sph.rho[j] * (P_sph.pres[j] - P_sph.pres[i]) * dwij
    # Sph - Bnd: include dummy pressure in the fluid pressure-gradient estimate.
    for bj in wp.hash_grid_query(grid_bnd, ri, sph_support):
        ribj = ri - P_bnd.pos[bj]
        r2 = wp.dot(ribj, ribj)
        if r2 > R2_MIN and r2 < sph_support * sph_support:
            dist = wp.sqrt(r2)
            dwibj = Kernel_dw_Wendland(dist, sph_h) * ribj / dist  # grad_i Wibj
            pgf = pgf - P_bnd.m[bj] / P_bnd.rho[bj] * (P_bnd.pres[bj] - P_sph.pres[i]) * dwibj
    # The reaction kernel divides by porosity, so retain the configured positive floor.
    P_sph.porosity[i] = wp.clamp(1.0 - solid, eps_min, eps_max)
    P_sph.pgf[i] = pgf


@wp.kernel
def Kernel_interaction_dem(P_sph: SPHptl, P_dem: DEMptl,
                           grid_sph: wp.uint64, grid_dem: wp.uint64,
                           mu: float,
                           eps_min: float, eps_max: float, dt: float) -> None:
    """
    Interpolate fluid fields and add pressure force / semi-implicit drag to DEM.

    Index convention: a is the DEM subject, b is a DEM neighbour and j is an
    SPH fluid neighbour. Dummy and DEM boundary particles do not enter interpolation.

    P_sph: Particle structure of SPH particles, including -grad(p) in pgf [N_sph]
    P_dem: Particle structure of moving DEM spheres [N_dem]
    grid_sph, grid_dem: HashGrid handles for current fluid / DEM positions
    P_dem.h[a]: this particle's coupling smoothing length; support is 2*h [m]
    mu: dynamic fluid viscosity [Pa s]
    eps_min, eps_max: allowed fluid-fraction bounds [-]
    dt: time interval for the semi-implicit drag coefficient [s]

    # Output
    P_dem.flt_s[a], P_dem.porosity[a]
    P_dem.drag[a], P_dem.pressure_force[a], P_dem.force[a] [N]
    Force is accumulated; the interpolation and individual fluid-force buffers are reset.

    No gravity here: the DEM contact stage already adds it once. Dry particles have
    zero pressure/drag, including a particle that has just left the fluid.
    """
    tid = wp.tid()
    a = wp.hash_grid_point_id(grid_dem, tid)
    ra = P_dem.pos[a]
    h = P_dem.h[a]
    support = 2.0 * h
    flt = float(0.0)
    solid = float(0.0)
    rho_f = float(0.0)
    vel_f = wp.vec3(0.0, 0.0, 0.0)
    pgf = wp.vec3(0.0, 0.0, 0.0)
    # Dem - Sph: volume-weighted sums for velocity, density and negative pressure gradient.
    for j in wp.hash_grid_query(grid_sph, ra, support):
        dist = wp.length(ra - P_sph.pos[j])
        if dist < support:
            Vw = P_sph.m[j] / P_sph.rho[j] * Kernel_w_Wendland(dist, h)  # Vj * Waj [-]
            flt = flt + Vw
            rho_f = rho_f + P_sph.rho[j] * Vw
            vel_f = vel_f + P_sph.vel[j] * Vw
            pgf = pgf + P_sph.pgf[j] * Vw
    # Dem - Dem: include a itself when estimating the local solid fraction.
    for b in wp.hash_grid_query(grid_dem, ra, support):
        dist = wp.length(ra - P_dem.pos[b])
        if dist < support:
            solid = solid + P_dem.volume[b] * Kernel_w_Wendland(dist, h)
    # Clear stored coupling loads so a sphere leaving the fluid cannot retain old drag.
    P_dem.flt_s[a] = flt
    P_dem.drag[a] = wp.vec3(0.0, 0.0, 0.0)
    P_dem.pressure_force[a] = wp.vec3(0.0, 0.0, 0.0)
    P_dem.porosity[a] = eps_max
    if flt > 1.0e-12:
        # Shepard normalization and porosity clamp at the moving DEM position.
        eps = wp.clamp(1.0 - solid / flt, eps_min, eps_max)
        urel = vel_f / flt - P_dem.vel[a]          # interpolated fluid - DEM velocity [m/s]
        beta = DEM_drag_beta(eps, rho_f / flt, mu, 2.0 * P_dem.radius[a], wp.length(urel))
        # Semi-implicit drag bounds the drag-only velocity increment by |urel|.
        coef = beta * P_dem.volume[a] / P_dem.m[a]  # velocity relaxation rate [1/s]
        drag = P_dem.m[a] * coef / (1.0 + dt * coef) * urel
        # Pressure force = sphere volume * interpolated (-grad p), without extra gravity.
        pressure_force = P_dem.m[a] / P_dem.rho[a] * pgf / flt
        P_dem.porosity[a] = eps
        P_dem.drag[a] = drag
        P_dem.pressure_force[a] = pressure_force
        P_dem.force[a] = P_dem.force[a] + drag + pressure_force  # add to contact + gravity


@wp.kernel
def Kernel_interaction_sph(P_sph: SPHptl, P_dem: DEMptl,
                           grid_sph: wp.uint64, grid_dem: wp.uint64,
                           support: float) -> None:
    """
    Add DEM drag reaction to the fluid acceleration before SPH integration.

    Index convention: i is the SPH subject and b is a moving DEM neighbour.

    P_sph: Particle structure of SPH particles, including current positive porosity [N_sph]
    P_dem: Particle structure with current DEM drag and flt_s values [N_dem]
    grid_sph, grid_dem: HashGrid handles for current fluid / DEM positions
    support: maximum 2*P_dem.h for the broad neighbour search [m]
    Each reaction uses the source DEM h, matching its interpolation weights.

    # Output
    P_sph.acc_dem[i] [m/s^2], P_sph.acc[i] (additive) [m/s^2]

    Reference reaction: -sum_b Fd_b W_ib / flt_b / (rho_i * eps_i).

    The reference volume-averaged convention balances drag against eps_i*m_i,
    with porosity held fixed during this exchange, rather than bare SPH mass.
    Pressure already enters the SPH equations; only drag is returned here.
    """
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid_sph, tid)
    ri = P_sph.pos[i]
    force_density = wp.vec3(0.0, 0.0, 0.0)        # distributed drag reaction [N/m^3]
    # Sph - Dem: distribute each wet DEM drag with its fluid-only normalization.
    for b in wp.hash_grid_query(grid_dem, ri, support):
        dist = wp.length(ri - P_dem.pos[b])
        if dist < 2.0 * P_dem.h[b] and P_dem.flt_s[b] > 1.0e-12:
            force_density = force_density - P_dem.drag[b] * Kernel_w_Wendland(dist, P_dem.h[b]) / P_dem.flt_s[b]
    # Convert reaction density to fluid acceleration with the reference porosity factor.
    acc = force_density / (P_sph.rho[i] * P_sph.porosity[i])
    P_sph.acc_dem[i] = acc
    P_sph.acc[i] = P_sph.acc[i] + acc             # retain the preceding SPH-only forces

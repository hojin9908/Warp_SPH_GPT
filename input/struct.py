import warp as wp

@wp.struct
class SPHptl:
    """
    - Structure of SPH (fluid) particles.
    - SoA structure
    - Read with "P.pos[i]"
    - SPH-DEM fields share the same SPH indices i / j.
    """
    pos: wp.array(dtype=wp.vec3)         # [N_sph,3]             Position                               [m]
    vel: wp.array(dtype=wp.vec3)         # [N_sph,3]             Velocity                               [m/s]
    rho_raw: wp.array(dtype=float)       # [N_sph]               Density (direct summation)             [kg/m^3]
    rho: wp.array(dtype=float)           # [N_sph]               Density (after Shepard filter)         [kg/m^3]
    pres: wp.array(dtype=float)          # [N_sph]               Pressure                               [Pa]
    acc: wp.array(dtype=wp.vec3)         # [N_sph,3]             Acceleration                           [m/s^2]
    m: wp.array(dtype=float)             # [N_sph]               Mass                                   [kg]
    flt: wp.array(dtype=float)           # [N_sph]               Shepard filter (SPH+BND)                 [-]
    porosity: wp.array(dtype=float)      # [N_sph]               Fluid fraction at SPH position         [-]
    pgf: wp.array(dtype=wp.vec3)         # [N_sph,3]             Negative pressure gradient             [Pa/m]
    acc_dem: wp.array(dtype=wp.vec3)     # [N_sph,3]             DEM drag reaction acceleration         [m/s^2]

@wp.struct
class BNDptl:
    """
    - Structure of boundary (dummy) particles.
    - SoA structure
    - Read with "B.pos[i]"
    """
    pos: wp.array(dtype=wp.vec3)         # [N_bnd,3]             Position                               [m]
    vel: wp.array(dtype=wp.vec3)         # [N_bnd,3]             Velocity                               [m/s]
    rho_raw: wp.array(dtype=float)       # [N_bnd]               Density (direct summation)             [kg/m^3]
    rho: wp.array(dtype=float)           # [N_bnd]               Density (after Shepard filter)         [kg/m^3]
    pres: wp.array(dtype=float)          # [N_bnd]               Pressure                               [Pa]
    acc: wp.array(dtype=wp.vec3)         # [N_bnd,3]             Acceleration                           [m/s^2]
    m: wp.array(dtype=float)             # [N_bnd]               Mass                                   [kg]


@wp.struct
class DEMptl:
    """
    - Structure of DEM (moving sphere) particles.
    - SoA structure. Read with "P_dem.pos[a]".
    - Index convention: a / b are the DEM subject / neighbour.
    - Position, velocity, force, angular velocity and torque use all three axes.
    - Mass, force, torque and inertia use 3D SI units.
    - Contact history uses explicit old/read and new/write CSR fields per
      contact kind, keyed by stable neighbour indices rather than grid order.
    """
    pos: wp.array(dtype=wp.vec3)         # [N_dem,3]             Position                               [m]
    vel: wp.array(dtype=wp.vec3)         # [N_dem,3]             Velocity                               [m/s]
    acc: wp.array(dtype=wp.vec3)         # [N_dem,3]             Acceleration (force / mass)            [m/s^2]
    force: wp.array(dtype=wp.vec3)       # [N_dem,3]             Total force                            [N]
    omega: wp.array(dtype=wp.vec3)       # [N_dem,3]             Angular velocity                       [rad/s]
    torque: wp.array(dtype=wp.vec3)      # [N_dem,3]             Contact torque                         [N m]
    radius: wp.array(dtype=float)        # [N_dem]               Sphere radius                          [m]
    rho: wp.array(dtype=float)           # [N_dem]               Material density                       [kg/m^3]
    m: wp.array(dtype=float)             # [N_dem]               Mass                                   [kg]
    volume: wp.array(dtype=float)        # [N_dem]               Sphere volume ((4/3) * pi * radius^3)  [m^3]
    inertia: wp.array(dtype=float)       # [N_dem]               Moment of inertia                      [kg m^2]
    K: wp.array(dtype=float)             # [N_dem]               Normal / tangential stiffness          [N/m]
    eta: wp.array(dtype=float)           # [N_dem]               Normal / tangential damping            [N s/m]
    mu: wp.array(dtype=float)            # [N_dem]               Coulomb friction coefficient           [-]
    # DEM-DEM CSR: *_old is read history; *_new is this step's write destination.
    contact_dem_offset_old: wp.array(dtype=wp.int32)   # [N_dem+1] previous/current CSR row offsets
    contact_dem_id_old: wp.array(dtype=wp.int32)       # [E_dem_old] stable neighbour IDs read this step
    tang_dem_old: wp.array(dtype=wp.vec3)              # [E_dem_old] tangential history read this step [m]
    contact_dem_offset_new: wp.array(dtype=wp.int32)   # [N_dem+1] contact counts, then new CSR offsets
    contact_dem_id_new: wp.array(dtype=wp.int32)       # [E_dem_new] stable neighbour IDs written this step
    tang_dem_new: wp.array(dtype=wp.vec3)              # [E_dem_new] updated tangential history [m]
    # DEM-boundary CSR follows the same old/read -> new/write -> swap lifecycle.
    contact_bnd_offset_old: wp.array(dtype=wp.int32)   # [N_dem+1] previous/current boundary CSR offsets
    contact_bnd_id_old: wp.array(dtype=wp.int32)       # [E_bnd_old] stable boundary IDs read this step
    tang_bnd_old: wp.array(dtype=wp.vec3)              # [E_bnd_old] boundary history read this step [m]
    contact_bnd_offset_new: wp.array(dtype=wp.int32)   # [N_dem+1] counts, then new boundary CSR offsets
    contact_bnd_id_new: wp.array(dtype=wp.int32)       # [E_bnd_new] stable boundary IDs written this step
    tang_bnd_new: wp.array(dtype=wp.vec3)              # [E_bnd_new] updated boundary history [m]
    flt_s: wp.array(dtype=float)         # [N_dem]               Shepard factor (fluid SPH only)        [-]
    porosity: wp.array(dtype=float)      # [N_dem]               Fluid fraction at DEM position         [-]
    drag: wp.array(dtype=wp.vec3)        # [N_dem,3]             Drag force                             [N]
    pressure_force: wp.array(dtype=wp.vec3)# [N_dem,3]             Pressure force                         [N]


@wp.struct
class DEMBNDptl:
    """
    - Structure of DEM boundary (fixed sphere) particles.
    - SoA structure. Read with "P_dem_bnd.pos[dbi]".
    - Index convention: dbi / dbj are the DEM boundary subject / neighbour.
    - Supplies fixed geometry and zero wall velocity/spin to the contact law.
    - Boundary force, torque and acceleration are neither stored nor integrated.
    """
    pos: wp.array(dtype=wp.vec3)         # [N_dem_bnd,3]         Fixed position                         [m]
    vel: wp.array(dtype=wp.vec3)         # [N_dem_bnd,3]         Wall velocity (zero)                   [m/s]
    omega: wp.array(dtype=wp.vec3)       # [N_dem_bnd,3]         Angular velocity (zero)                [rad/s]
    radius: wp.array(dtype=float)        # [N_dem_bnd]           Boundary sphere radius                 [m]
    rho: wp.array(dtype=float)           # [N_dem_bnd]           Material density                       [kg/m^3]
    m: wp.array(dtype=float)             # [N_dem_bnd]           Mass                                   [kg]
    volume: wp.array(dtype=float)        # [N_dem_bnd]           Boundary sphere volume                 [m^3]
    inertia: wp.array(dtype=float)       # [N_dem_bnd]           Moment of inertia                      [kg m^2]
    K: wp.array(dtype=float)             # [N_dem_bnd]           Normal / tangential stiffness          [N/m]
    eta: wp.array(dtype=float)           # [N_dem_bnd]           Normal / tangential damping            [N s/m]
    mu: wp.array(dtype=float)            # [N_dem_bnd]           Coulomb friction coefficient           [-]

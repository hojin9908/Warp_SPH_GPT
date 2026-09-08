import warp as wp

@wp.struct
class SPHptl:
    """
    - Structure of SPH (fluid) particles.
    - SoA structure
    - Read with "P.pos[i]"
    """
    pos: wp.array(dtype=wp.vec3)        # [N_sph,3]     Position                        [m]
    vel: wp.array(dtype=wp.vec3)        # [N_sph,3]     Velocity                        [m/s]
    rho_raw: wp.array(dtype=float)      # [N_sph]       Density (direct summation)      [kg/m^3]
    rho: wp.array(dtype=float)          # [N_sph]       Density (after Shepard filter)  [kg/m^3]
    pres: wp.array(dtype=float)         # [N_sph]       Pressure                        [Pa]
    acc: wp.array(dtype=wp.vec3)        # [N_sph,3]     Acceleration                    [m/s^2]
    m: wp.array(dtype=float)            # [N_sph]       Mass                            [kg]
    flt: wp.array(dtype=float)          # [N_sph]       Shepard filter (SPH+BND)   

@wp.struct
class BNDptl:
    """
    - Structure of boundary (dummy) particles.
    - SoA structure
    - Read with "B.pos[i]"
    """
    pos: wp.array(dtype=wp.vec3)        # [N_bnd,3]     Position                        [m]
    vel: wp.array(dtype=wp.vec3)        # [N_bnd,3]     Velocity                        [m/s]
    rho_raw: wp.array(dtype=float)      # [N_bnd]       Density (direct summation)      [kg/m^3]
    rho: wp.array(dtype=float)          # [N_bnd]       Density (after Shepard filter)  [kg/m^3]
    pres: wp.array(dtype=float)         # [N_bnd]       Pressure                        [Pa]
    acc: wp.array(dtype=wp.vec3)        # [N_bnd,3]     Acceleration                    [m/s^2]
    m: wp.array(dtype=float)            # [N_bnd]       Mass                            [kg]   

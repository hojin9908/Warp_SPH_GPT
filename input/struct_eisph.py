import warp as wp


@wp.struct
class EISPHptl:
    """
    - Structure of EISPH fluid or ghost boundary particles.
    - SoA structure. Read with "P_sph.pos[i]" or "P_bnd.pos[bi]".
    - The same structure is used for fluid and boundary particles.
    - Index convention: i / j are fluid, bi / bj are ghost boundary particles.
    - Fluid positions are fixed throughout the Eulerian simulation.
    """

    pos: wp.array(dtype=wp.vec3)         # [N_eisph,3] Position (fixed)                    [m]
    vel: wp.array(dtype=wp.vec3)         # [N_eisph,3] Corrected velocity                 [m/s]
    vel_star: wp.array(dtype=wp.vec3)    # [N_eisph,3] Predictor velocity                 [m/s]
    vel_bc: wp.array(dtype=wp.vec3)      # [N_eisph,3] Prescribed Dirichlet velocity      [m/s]
    acc: wp.array(dtype=wp.vec3)         # [N_eisph,3] Velocity increment / dt            [m/s^2]
    kgc: wp.array(dtype=wp.mat22)        # [N_eisph,2,2] Kernel-gradient correction       [-]
    rho: wp.array(dtype=float)           # [N_eisph] Reference density                    [kg/m^3]
    m: wp.array(dtype=float)             # [N_eisph] Particle mass per unit depth         [kg/m]
    pres: wp.array(dtype=float)          # [N_eisph] Current pressure                     [Pa]
    Aij: wp.array(dtype=float)           # [N_eisph] Sum of signed PDF PPE coefficients  [1/m^2]
    bi: wp.array(dtype=float)            # [N_eisph] Pressure-Poisson right-hand side     [Pa/m^2]
    Aijpj: wp.array(dtype=float)         # [N_eisph] Sum of Aij times old neighbour p     [Pa/m^2]
    divergence: wp.array(dtype=float)    # [N_eisph] Velocity divergence                  [1/s]
    residual: wp.array(dtype=float)      # [N_eisph] PPE residual; [0] bi-sum scratch     [Pa/m^2]
    mirror: wp.array(dtype=wp.int32)     # [N_eisph] Stable mirrored fluid index          [-]

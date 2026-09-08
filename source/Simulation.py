import warp as wp

from input.Config import Solv
from input.struct import SPHptl, BNDptl

from kernel.KERNEL_rho import Kernel_shepard_sph, Kernel_density_sph, Kernel_density_bnd
from kernel.KERNEL_pres import Kernel_pres_sph, Kernel_pres_bnd
from kernel.KERNEL_force import Kernel_force_sph
from kernel.KERNEL_step import Kernel_step_sph

def SPH_OneStep(solv: Solv,
                P_sph: SPHptl,
                P_bnd: BNDptl,
                grid_sph: wp.HashGrid,
                grid_bnd: wp.HashGrid,
                step: int):
    """
    Simulate one step of SPH.

    shepard filter -> density -> pressure -> force -> integration

    solv: property of the simulation
    P_sph: Particle structure of SPH particles      [N_sph]
    P_bnd: Particle structure of BND particles      [N_bnd]
    grid_sph: HashGrid for SPH particles
    grid_bnd: HashGrid for BND particles
    step: current step of the simulation
    """
    n_sph = P_sph.pos.shape[0]
    n_bnd = P_bnd.pos.shape[0]
    use_shepard = solv.shepard_step > 0 and step % solv.shepard_step == 0

    # 1) shepard filter
    # P_sph.flt is refreshed only every solv.shepard_step steps and is kept in
    # between, so Kernel_density_sph divides by the filter computed at the last
    # refresh. Holding the normalization factor fixed is intended: it keeps the
    # correction consistent where the kernel support is truncated and the SPH
    # interpolation would otherwise underestimate the density.
    if use_shepard:
        wp.launch(
            Kernel_shepard_sph,
            dim = n_sph,
            inputs = [P_sph, P_bnd, grid_sph.id, grid_bnd.id, solv.support, solv.h]
        )

    # 2) density
    wp.launch(
        Kernel_density_sph,
        dim = n_sph,
        inputs = [P_sph, P_bnd, grid_sph.id, grid_bnd.id, solv.support, solv.h]
    )

    wp.launch(
        Kernel_density_bnd,
        dim = n_bnd,
        inputs = [P_sph, P_bnd, grid_sph.id, grid_bnd.id, solv.support, solv.h]
    )

    # 3) pressure (Tait EOS)
    wp.launch(
        Kernel_pres_sph,
        dim = n_sph,
        inputs = [P_sph, solv.rho0, solv.c0, solv.gamma]
    )
    wp.launch(
        Kernel_pres_bnd,
        dim = n_bnd,
        inputs = [P_bnd, solv.rho0, solv.c0, solv.gamma]
    )

    # 4) force (pressure + viscosity + gravity)
    wp.launch(
        Kernel_force_sph,
        dim = n_sph,
        inputs = [P_sph, P_bnd, grid_sph.id, grid_bnd.id,
                  solv.support, solv.h, solv.mu, solv.g]
    )

    # 5) integration (semi-implicit Euler)
    wp.launch(
        Kernel_step_sph,
        dim = n_sph,
        inputs = [P_sph, solv.dt]
    )

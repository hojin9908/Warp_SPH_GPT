import warp as wp

from input.Config_SPH_DEM import Solv
from input.input_reader import dem_search_scales
from input.struct import SPHptl, BNDptl, DEMptl, DEMBNDptl

from kernel.KERNEL_rho import Kernel_shepard_sph, Kernel_density_sph, Kernel_density_bnd
from kernel.KERNEL_pres import Kernel_pres_sph, Kernel_pres_bnd
from kernel.KERNEL_force import Kernel_force_sph
from kernel.KERNEL_step import Kernel_step_sph
from kernel.KERNEL_DEM_force import Kernel_count_dem_contacts, Kernel_force_dem
from kernel.KERNEL_DEM_BC import Kernel_count_bnd_contacts, Kernel_bc_dem
from kernel.KERNEL_DEM_step import Kernel_step_dem
from kernel.KERNEL_SPHDEM_interaction import (
    Kernel_prep_sphdem, Kernel_interaction_dem, Kernel_interaction_sph)

def SPH_OneStep(solv: Solv,
                P_sph: SPHptl,
                P_bnd: BNDptl,
                grid_sph: wp.HashGrid,
                grid_bnd: wp.HashGrid,
                step: int,
                integrate: bool = True):
    """
    Simulate one step of SPH.

    shepard filter -> density -> pressure -> force -> integration

    solv: property of the simulation
    P_sph: Particle structure of SPH particles      [N_sph]
    P_bnd: Particle structure of BND particles      [N_bnd]
    grid_sph: HashGrid for SPH particles
    grid_bnd: HashGrid for BND particles
    step: current step of the simulation
    integrate: False leaves positions / velocities unchanged so DEM coupling
        can add its reaction before the shared final integration

    # Output
    P_sph.flt, P_sph.rho, P_bnd.rho, P_sph.pres, P_bnd.pres, P_sph.acc
    P_sph.vel and P_sph.pos are updated only when integrate is True.
    """
    n_sph = P_sph.pos.shape[0]
    n_bnd = P_bnd.pos.shape[0]
    # The density kernel divides by flt, so initialize it even if periodic filtering is off.
    use_shepard = step == 0 or (solv.shepard_step > 0 and step % solv.shepard_step == 0)

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
    if integrate:
        wp.launch(
            Kernel_step_sph,
            dim = n_sph,
            inputs = [P_sph, solv.dt]
        )


def SPHDEM_OneStep(solv: Solv,
                   P_sph: SPHptl, P_bnd: BNDptl,
                   P_dem: DEMptl, P_dem_bnd: DEMBNDptl,
                   grid_sph: wp.HashGrid, grid_bnd: wp.HashGrid,
                   grid_dem: wp.HashGrid, grid_dem_bnd: wp.HashGrid,
                   step: int,
                   search_scales: tuple[float, float, float, float] | None = None):
    """
    Simulate one coupled SPH-DEM step at a common position / velocity time level.

    SPH fields / forces -> DEM contacts -> fluid exchange -> SPH / DEM integration

    solv: property of the simulation, material coefficients and time step
    P_sph: Particle structure of SPH particles [N_sph]
    P_bnd: Particle structure of SPH dummy boundary particles [N_bnd]
    P_dem: Particle structure of moving DEM spheres [N_dem]
    P_dem_bnd: Particle structure of fixed DEM boundary spheres [N_dem_bnd]
    grid_sph, grid_bnd: HashGrid for fluid / dummy boundary particles
    grid_dem, grid_dem_bnd: HashGrid for moving / fixed DEM particles
    step: current zero-based simulation step

    The caller builds moving grids each step and fixed grids once at startup.
    Original SPH density, pressure, force and integration kernels are reused.

    # Output
    Updated SPH fields (including porosity, pgf, acc_dem) and DEM loads / histories.
    Moving SPH / DEM positions and velocities, plus DEM angular velocities.
    Fixed DEM boundary geometry and zero wall velocity/spin remain unchanged.
    """
    n_sph = P_sph.pos.shape[0]
    n_dem = P_dem.pos.shape[0]
    radius_max, boundary_radius_max, h_max, h_uniform = (
        dem_search_scales(P_dem, P_dem_bnd) if search_scales is None else search_scales)
    # 1) SPH density, pressure and forces; leave motion at the current time level.
    SPH_OneStep(solv, P_sph, P_bnd, grid_sph, grid_bnd, step, integrate=False)

    # 2) Count and scan the new DEM-DEM CSR rows.
    # Before calculation:
    #   *_old = previous step's readable contact CSR
    #   *_new = this step's writable contact CSR
    P_dem.contact_dem_offset_new.zero_()
    wp.launch(Kernel_count_dem_contacts, dim=n_dem,
              inputs=[P_dem, grid_dem.id, radius_max])
    # Before scan, contact_dem_offset_new[a] is the number of particles
    # contacting P_dem[a]. The last element remains zero.
    # After scan, [offset_new[a], offset_new[a+1]) is P_dem[a]'s CSR row.
    wp.utils.array_scan(P_dem.contact_dem_offset_new,
                        P_dem.contact_dem_offset_new, inclusive=False)
    # The last CSR offset is the total number of directed DEM contacts E_dem.
    n_contact_dem = int(
        P_dem.contact_dem_offset_new[n_dem:n_dem + 1].numpy()[0])
    if n_contact_dem < 0:
        raise OverflowError("directed DEM contact count exceeded int32 CSR capacity")

    # Allocate exactly one ID and one tangential-history vector per new edge.
    P_dem.contact_dem_id_new = wp.empty(
        n_contact_dem, dtype=wp.int32, device=P_dem.pos.device)
    P_dem.tang_dem_new = wp.empty(
        n_contact_dem, dtype=wp.vec3, device=P_dem.pos.device)
    wp.launch(Kernel_force_dem, dim=n_dem,
              inputs=[P_dem, grid_dem.id, radius_max, solv.g, solv.dt])
    # The finished write side becomes the readable side for the next step.
    # The retired old arrays move to *_new, which also keeps their CUDA memory
    # alive until later same-stream work has finished using them.
    P_dem.contact_dem_offset_old, P_dem.contact_dem_offset_new = (
        P_dem.contact_dem_offset_new, P_dem.contact_dem_offset_old)
    P_dem.contact_dem_id_old, P_dem.contact_dem_id_new = (
        P_dem.contact_dem_id_new, P_dem.contact_dem_id_old)
    P_dem.tang_dem_old, P_dem.tang_dem_new = (
        P_dem.tang_dem_new, P_dem.tang_dem_old)

    # 3) Count and scan the new moving-DEM-to-fixed-boundary contact CSR.
    # The wall is kinematic: only the force/torque on moving DEM is evaluated.
    P_dem.contact_bnd_offset_new.zero_()
    wp.launch(Kernel_count_bnd_contacts, dim=n_dem,
              inputs=[P_dem, P_dem_bnd, grid_dem_bnd.id,
                      boundary_radius_max])
    # Before scan this is the contact count of each moving particle; after
    # scan it stores the new DEM-boundary CSR row range for that particle.
    wp.utils.array_scan(P_dem.contact_bnd_offset_new,
                        P_dem.contact_bnd_offset_new, inclusive=False)
    # The final offset is the total number of directed boundary contacts E_bnd.
    n_contact_bnd = int(
        P_dem.contact_bnd_offset_new[n_dem:n_dem + 1].numpy()[0])
    if n_contact_bnd < 0:
        raise OverflowError("DEM-boundary contact count exceeded int32 CSR capacity")

    P_dem.contact_bnd_id_new = wp.empty(
        n_contact_bnd, dtype=wp.int32, device=P_dem.pos.device)
    P_dem.tang_bnd_new = wp.empty(
        n_contact_bnd, dtype=wp.vec3, device=P_dem.pos.device)
    wp.launch(Kernel_bc_dem, dim=n_dem,
              inputs=[P_dem, P_dem_bnd, grid_dem_bnd.id,
                      boundary_radius_max, solv.dt])
    # Publish the completed boundary CSR and retain the retired buffers in
    # *_new until they are safely reused on the next step.
    P_dem.contact_bnd_offset_old, P_dem.contact_bnd_offset_new = (
        P_dem.contact_bnd_offset_new, P_dem.contact_bnd_offset_old)
    P_dem.contact_bnd_id_old, P_dem.contact_bnd_id_new = (
        P_dem.contact_bnd_id_new, P_dem.contact_bnd_id_old)
    P_dem.tang_bnd_old, P_dem.tang_bnd_new = (
        P_dem.tang_bnd_new, P_dem.tang_bnd_old)

    # 4) Fluid fraction and pressure gradient on the SPH particle positions.
    wp.launch(Kernel_prep_sphdem, dim=n_sph,
              inputs=[P_sph, P_bnd, P_dem, grid_sph.id, grid_bnd.id, grid_dem.id,
                      2.0 * h_max, h_uniform, solv.support, solv.h,
                      solv.dem_porosity_min, solv.dem_porosity_max])
    # 5) Fluid interpolation -> DEM pressure force and semi-implicit drag.
    wp.launch(Kernel_interaction_dem, dim=n_dem,
              inputs=[P_sph, P_dem, grid_sph.id, grid_dem.id,
                      solv.mu,
                      solv.dem_porosity_min, solv.dem_porosity_max, solv.dt])
    # 6) Distribute the stored DEM drag reaction to SPH using the same support.
    wp.launch(Kernel_interaction_sph, dim=n_sph,
              inputs=[P_sph, P_dem, grid_sph.id, grid_dem.id, 2.0 * h_max])

    # 7) Integrate moving particles only, after both sides of the exchange are ready.
    wp.launch(Kernel_step_sph, dim=n_sph, inputs=[P_sph, solv.dt])
    wp.launch(Kernel_step_dem, dim=n_dem, inputs=[P_dem, solv.dt])

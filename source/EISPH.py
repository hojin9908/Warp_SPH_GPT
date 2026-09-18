from collections.abc import Callable

import numpy as np
import warp as wp

from input.Config import Solv
from input.input_reader import validate_eisph_particles
from input.struct_eisph import EISPHptl
from kernel.KERNEL_EISPH_KNL import Kernel_prepare_kgc
from kernel.KERNEL_EISPH_BC import (
    Kernel_Dirichlet_BC,
    Kernel_pressure_boundary_eisph,
)
from kernel.KERNEL_EISPH_force import Kernel_advec_vis_eisph
from kernel.KERNEL_EISPH_PPE import (
    Kernel_build_ppe_eisph,
    Kernel_remove_bi_mean,
    Kernel_solve_ppe_eisph,
)
from kernel.KERNEL_EISPH_step import (
    Kernel_correct_eisph,
    Kernel_diagnostics_eisph,
)


def EISPH_OneStep(solv: Solv,
                  P_sph: EISPHptl,
                  P_bnd: EISPHptl,
                  grid_sph: wp.HashGrid,
                  grid_bnd: wp.HashGrid,
                  diagnostics: bool = True) -> None:
    """
    Simulate one step of Eulerian incompressible SPH.

    ghost velocity -> non-pressure predictor -> pressure-Poisson equation
    -> velocity projection -> optional diagnostics

    solv: EISPH cavity properties, material coefficients and time step
    P_sph: EISPHptl structure of fixed fluid points [N_sph]
    P_bnd: EISPHptl structure of fixed ghost boundary points [N_bnd]
    grid_sph: HashGrid built once from P_sph.pos
    grid_bnd: HashGrid built once from P_bnd.pos
    diagnostics: calculate divergence and PPE residual after projection

    Index convention: i / j are the fluid subject and neighbour,
    bi / bj are the ghost boundary subject and neighbour.

    # Output
    Update predictor velocity, PPE fields, pressure and corrected velocity in
    P_sph. Update diagnostics when requested. Update ghost fields in P_bnd.
    P_sph.pos and P_bnd.pos remain unchanged.
    """
    n_sph = P_sph.pos.shape[0]
    n_bnd = P_bnd.pos.shape[0]

    # 1) Apply the prescribed velocity Dirichlet BC to the current velocity.
    wp.launch(Kernel_Dirichlet_BC, dim=n_bnd,
              inputs=[P_sph, P_bnd, 0])
    # 2) Explicit Eulerian convection and viscosity predictor.
    wp.launch(Kernel_advec_vis_eisph, dim=n_sph,
              inputs=[P_sph, P_bnd, grid_sph.id, grid_bnd.id,
                      solv.support, solv.h, solv.nu, solv.dt])
    # 3) Use the predictor velocity in the pressure-Poisson boundary terms.
    wp.launch(Kernel_Dirichlet_BC, dim=n_bnd,
              inputs=[P_sph, P_bnd, 1])
    # 4) Assemble PDF-signed sum(Aij), bi and sum(Aij*p_j^t).
    wp.launch(Kernel_build_ppe_eisph, dim=n_sph,
              inputs=[P_sph, P_bnd, grid_sph.id, grid_bnd.id,
                      solv.support, solv.h, solv.rho0, solv.dt])
    # A closed Neumann problem is solvable only when the RHS sum is zero.
    # residual[0] is temporary storage for Warp's parallel reduction result.
    wp.utils.array_sum(P_sph.bi, out=P_sph.residual[:1])
    wp.launch(Kernel_remove_bi_mean, dim=n_sph, inputs=[P_sph, n_sph])

    # 5) One implicit diagonal pressure update; no iterative PPE loop.
    wp.launch(Kernel_solve_ppe_eisph, dim=n_sph, inputs=[P_sph])

    # 6) Neumann ghost pressure followed by the velocity projection.
    wp.launch(Kernel_pressure_boundary_eisph, dim=n_bnd,
              inputs=[P_sph, P_bnd])
    wp.launch(Kernel_correct_eisph, dim=n_sph,
              inputs=[P_sph, P_bnd, grid_sph.id, grid_bnd.id,
                      solv.support, solv.h, solv.dt])
    # 7) Publish corrected ghost velocity and optional diagnostic fields.
    wp.launch(Kernel_Dirichlet_BC, dim=n_bnd,
              inputs=[P_sph, P_bnd, 0])
    if diagnostics:
        wp.launch(Kernel_diagnostics_eisph, dim=n_sph,
                  inputs=[P_sph, P_bnd, grid_sph.id, grid_bnd.id,
                          solv.support, solv.h])


def run_eisph(solv: Solv,
              P_sph: EISPHptl,
              P_bnd: EISPHptl,
              progress: Callable[[int, int], None] | None = None
              ) -> tuple[np.ndarray, list[np.ndarray], list[float]]:
    """
    Run the lid-driven cavity and collect host velocity frames.

    solv: EISPH cavity and output configuration
    P_sph: initialized fluid EISPHptl [N_sph]
    P_bnd: initialized ghost boundary EISPHptl [N_bnd]
    progress: optional callback receiving (completed_step, total_step)

    Both point sets are fixed. Their HashGrid and kernel-gradient correction
    are built once before time integration and reused for every step.
    Diagnostics are evaluated only at output steps and the final step.

    # Output
    return: fixed positions [N_sph,3], velocity frames [frame][N_sph,3],
        and physical frame times [frame]. Particle fields are also updated in place.
    """
    validate_eisph_particles(solv, P_sph, P_bnd)
    with wp.ScopedDevice(solv.device):
        n_sph = P_sph.pos.shape[0]
        grid_sph = wp.HashGrid(solv.grid_slice, solv.grid_slice, solv.grid_slice)
        grid_bnd = wp.HashGrid(solv.grid_slice, solv.grid_slice, solv.grid_slice)

        # Both point sets are fixed, so their neighbour grids and KGC are built once.
        grid_sph.build(points=P_sph.pos, radius=solv.support)
        grid_bnd.build(points=P_bnd.pos, radius=solv.support)
        wp.launch(Kernel_prepare_kgc, dim=n_sph,
                  inputs=[P_sph, P_bnd, grid_sph.id, grid_bnd.id,
                          solv.support, solv.h])

        positions = P_sph.pos.numpy().copy()
        velocity_frames = [P_sph.vel.numpy().copy()]
        times = [0.0]

        for step in range(1, solv.n_steps + 1):
            store = (solv.output_step > 0 and step % solv.output_step == 0) or step == solv.n_steps
            EISPH_OneStep(solv, P_sph, P_bnd, grid_sph, grid_bnd,
                          diagnostics=store)
            if store:
                velocity_frames.append(P_sph.vel.numpy().copy())
                times.append(step * solv.dt)
                if progress is not None:
                    progress(step, solv.n_steps)

        return positions, velocity_frames, times

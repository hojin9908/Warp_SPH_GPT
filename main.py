import argparse
from typing import Any

import warp as wp

from input.Config_SPH_DEM import Solv
from input.struct import SPHptl, BNDptl, DEMptl, DEMBNDptl
from input.input_reader import load_sph_particles, validate_dem_particles
from source.Simulation import SPH_OneStep, SPHDEM_OneStep
from output.frame_writer import FrameWriter


def parsing() -> dict[str, Any]:
    """
    Read command-line overrides for the Solv configuration.

    Omitted optional arguments use argparse.SUPPRESS so that Solv keeps its
    configured defaults. --no-dem selects the SPH-only path.

    return: keyword arguments passed to Solv(**args)
    """
    parser = argparse.ArgumentParser(description="Warp 3D SPH-DEM Dam Break Solver")

    parser.add_argument("--device", type=str, help="device to use", default=argparse.SUPPRESS)
    parser.add_argument("--input-dir", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--steps", dest="n_steps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--dt", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--output-step", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--no-gif", dest="gif_save", action="store_false", default=argparse.SUPPRESS)
    parser.add_argument("--no-dem", dest="dem_enable", action="store_false", default=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--animation-dir", type=str, default=argparse.SUPPRESS)

    args = vars(parser.parse_args())
    return args


def run_forward(solv: Solv,
                P_sph: SPHptl,
                P_bnd: BNDptl,
                grid_sph: wp.HashGrid,
                grid_bnd: wp.HashGrid,
                P_dem: DEMptl = None,
                P_dem_bnd: DEMBNDptl = None,
                grid_dem: wp.HashGrid = None,
                grid_dem_bnd: wp.HashGrid = None) -> None:
    """
    Run forward SPH or coupled SPH-DEM simulation and collect particle outputs.

    solv: configuration of solv file
    P_sph: Particle structure of SPH particles      [N_sph]
    P_bnd: Particle structure of BND particles      [N_bnd]
    grid_sph: HashGrid for SPH particles
    grid_bnd: HashGrid for BND particles
    P_dem: Particle structure of moving DEM spheres [N_dem], required in coupled mode
    P_dem_bnd: Particle structure of fixed DEM spheres [N_dem_bnd], required in coupled mode
    grid_dem, grid_dem_bnd: HashGrid for moving / fixed DEM spheres

    # Output
    Particle structures are updated in place. If solv.output_step > 0, write
    separate SPH / DEM VTP and PVD series and, if requested, a combined GIF.
    The initial and final states are included even outside the regular cadence.
    """
    print(f"\n[forward] {solv.n_steps} step \t\t (t={solv.n_steps*solv.dt:.3f} s\tdt={solv.dt:.1e} s)\n")
    # Fixed SPH and DEM boundary grids are built once because neither wall moves.
    grid_bnd.build(points=P_bnd.pos, radius=solv.support)
    coupled = solv.dem_enable and P_dem is not None
    if solv.dem_enable and not coupled:
        raise ValueError("DEM is enabled: provide DEM particles, boundaries and coupling grids")
    search_scales = None
    if coupled:
        search_scales = validate_dem_particles(solv, P_dem, P_dem_bnd)
        grid_dem_bnd.build(points=P_dem_bnd.pos, radius=max(2.0 * search_scales[1], solv.support))

    # The output module owns frame validation, VTK references and optional GIF data.
    writer = FrameWriter(solv, P_sph, P_bnd,
                         P_dem if coupled else None,
                         P_dem_bnd if coupled else None)

    if solv.output_step > 0:
        writer.write_frame(0)

    # Moving particles require rebuilt neighbour grids before every force evaluation.
    for step in range(solv.n_steps):
        grid_sph.build(points=P_sph.pos, radius=solv.support)
        if coupled:
            grid_dem.build(points=P_dem.pos, radius=max(2.0 * search_scales[0], 2.0 * search_scales[2]))
            SPHDEM_OneStep(solv, P_sph, P_bnd, P_dem, P_dem_bnd,
                          grid_sph, grid_bnd, grid_dem, grid_dem_bnd, step, search_scales)
        else:
            SPH_OneStep(solv, P_sph, P_bnd, grid_sph, grid_bnd, step)

        # the state after this step belongs to step+1
        if solv.output_step > 0 and ((step + 1) % solv.output_step == 0 or step + 1 == solv.n_steps):
            writer.write_frame(step + 1)
            print(f"[output] step {step+1:>6d} / {solv.n_steps}\t t={(step+1)*solv.dt:.4f} s")

    writer.finalize()


def main() -> None:
    """
    Initialize configuration, particles and grids, then run the forward solver.

    All Warp allocation, grid building and kernel launches use solv.device.
    DEM structures and grids are created only when solv.dem_enable is True.

    # Output
    Simulation results are written by run_forward according to the Solv settings.
    """
    args = parsing()
    solv = Solv(**args)
    wp.init()
    # Scope also controls HashGrid allocation and kernel launches in helper functions.
    with wp.ScopedDevice(solv.device):
        P_sph, P_bnd, P_dem, P_dem_bnd = load_sph_particles(solv)
        grid_sph = wp.HashGrid(solv.grid_slice, solv.grid_slice, solv.grid_slice)
        grid_bnd = wp.HashGrid(solv.grid_slice, solv.grid_slice, solv.grid_slice)
        if solv.dem_enable:
            # Allocate moving / fixed DEM structures; fluid fields belong to P_sph.
            grid_dem = wp.HashGrid(solv.grid_slice, solv.grid_slice, solv.grid_slice)
            grid_dem_bnd = wp.HashGrid(solv.grid_slice, solv.grid_slice, solv.grid_slice)
            run_forward(solv, P_sph, P_bnd, grid_sph, grid_bnd,
                        P_dem, P_dem_bnd, grid_dem, grid_dem_bnd)
        else:
            run_forward(solv, P_sph, P_bnd, grid_sph, grid_bnd)


if __name__ == "__main__":
    main()

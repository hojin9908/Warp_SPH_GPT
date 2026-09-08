import argparse
from typing import Any

import numpy as np
import warp as wp

from input.Config import Solv
from input.struct import SPHptl, BNDptl
from input.gen_ptl import DamPtlGeneration
from source.Simulation import SPH_OneStep
from output.output import save_vtk, save_pvd
from output.gif_gen import collect_frame, save_gif


def parsing() -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Warp SPH Solver")

    parser.add_argument("--device", type=str, help="device to use", default="cuda:0")

    args = vars(parser.parse_args())
    return args


def run_forward(solv: Solv,
                P_sph: SPHptl,
                P_bnd: BNDptl,
                grid_sph: wp.HashGrid,
                grid_bnd: wp.HashGrid) -> None:
    """
    Run Forward SPH Simulation

    solv: configuration of solv file
    P_sph: Particle structure of SPH particles      [N_sph]
    P_bnd: Particle structure of BND particles      [N_bnd]
    grid_sph: HashGrid for SPH particles
    grid_bnd: HashGrid for BND particles
    """
    print(f"\n[forward] {solv.n_steps} step \t\t (t={solv.n_steps*solv.dt:.3f} s\tdt={solv.dt:.1e} s)\n")
    grid_bnd.build(points=P_bnd.pos, radius=solv.support)

    # [(file name, t), ...] of every frame written, collected for the .pvd
    frames: list[tuple[str, float]] = []
    # host side copies of the same frames, kept only when a gif is wanted
    gif_frames: list[dict] = []
    if solv.output_step > 0:
        frames.append(save_vtk(P_sph, P_bnd, 0, 0.0))
        if solv.gif_save:
            gif_frames.append(collect_frame(P_sph, P_bnd, 0.0))

    for step in range(solv.n_steps):
        grid_sph.build(points=P_sph.pos, radius=solv.support)
        SPH_OneStep(solv, P_sph, P_bnd, grid_sph, grid_bnd, step)

        # the state after this step belongs to step+1
        if solv.output_step > 0 and (step + 1) % solv.output_step == 0:
            frames.append(save_vtk(P_sph, P_bnd, step + 1, (step + 1) * solv.dt))
            if solv.gif_save:
                gif_frames.append(collect_frame(P_sph, P_bnd, (step + 1) * solv.dt))
            print(f"[output] step {step+1:>6d} / {solv.n_steps}\t t={(step+1)*solv.dt:.4f} s")

    if gif_frames:
        gif = save_gif(gif_frames, solv)
        print(f"\n[output] {len(gif_frames)} frames -> {gif}")

    if frames:
        path = save_pvd(frames)
        print(f"\n[output] {len(frames)} frames -> {path}\n")
    



def main() -> None:
    args = parsing()
    solv = Solv(**args)
    wp.init()
    dam_generater = DamPtlGeneration(solv=solv)

    P_sph, P_bnd = dam_generater.build()            # [N_sph, SPHptl], [N_bnd, BNDptl]
    grid_sph = wp.HashGrid(solv.grid_slice, solv.grid_slice, 1)
    grid_bnd = wp.HashGrid(solv.grid_slice, solv.grid_slice, 1)

    run_forward(solv, P_sph, P_bnd, grid_sph, grid_bnd)


if __name__ == "__main__":
    main()

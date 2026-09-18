import argparse
from typing import Any

import warp as wp

from input.Config_EISPH import EISPHConfig
from input.gen_eisph import CavityPtlGeneration
from output.gif_eisph import save_cavity_gif
from source.EISPH import run_eisph


def parsing() -> dict[str, Any]:
    """
    Read command-line overrides for the EISPH configuration.

    Omitted optional arguments use argparse.SUPPRESS so EISPHConfig keeps its
    configured defaults.

    return: keyword arguments passed to EISPHConfig(**args)
    """
    parser = argparse.ArgumentParser(description="Eulerian ISPH lid-driven cavity")
    parser.add_argument("--device", type=str, help="device to use",
                        default=argparse.SUPPRESS)
    parser.add_argument("--steps", dest="n_steps", type=int,
                        default=argparse.SUPPRESS)
    parser.add_argument("--dt", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--dx", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--reynolds", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--output-step", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--fps", dest="gif_fps", type=int,
                        default=argparse.SUPPRESS)
    parser.add_argument("--animation", dest="animation_path", type=str,
                        default=argparse.SUPPRESS)
    return vars(parser.parse_args())


def main() -> None:
    """
    Initialize the fixed EISPH cavity, run it and save its GIF.

    All Warp allocation, grid building and kernel launches use solv.device.

    # Output
    animation/lid_driven_cavity.gif, or the path selected by --animation.
    """
    solv = EISPHConfig(**parsing())
    wp.init()
    # Scope controls particle allocation, HashGrid construction and kernel launches.
    with wp.ScopedDevice(solv.device):
        P_sph, P_bnd = CavityPtlGeneration(solv).build()
        positions, velocities, times = run_eisph(
            solv, P_sph, P_bnd,
            progress=lambda step, total: print(f"[EISPH] {step:>5d} / {total}"),
        )
        # Rendering consumes only fixed positions and stored host velocity frames.
        animation = save_cavity_gif(positions, velocities, times, solv)
    print(f"[EISPH] animation: {animation}")


if __name__ == "__main__":
    main()

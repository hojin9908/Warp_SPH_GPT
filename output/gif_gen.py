import os

import matplotlib
matplotlib.use("Agg")               # no display, the run is head-less

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from input.Config import Solv
from output.output import SPH_TYPE, gather_state
from input.struct import SPHptl, BNDptl

ANI_DIR = "animation"


def collect_frame(P_sph: SPHptl,
                  P_bnd: BNDptl,
                  t: float) -> dict:
    """
    Copy one frame off the device and keep only what the animation draws.

    Called at the same interval the vtk frames are written, so the gif and
    the ParaView time series show the same instants.

    P_sph: Particle structure of SPH particles      [N_sph]
    P_bnd: Particle structure of BND particles      [N_bnd]
    t: physical time of this frame [s]

    return: dict held in host memory until save_gif draws it
    """
    d = gather_state(P_sph, P_bnd)
    fluid = d["type"] == SPH_TYPE
    return {
        "t": t,
        "pos_sph": d["pos"][fluid, :2],
        "pos_bnd": d["pos"][~fluid, :2],
        "vel_sph": np.linalg.norm(d["vel"][fluid], axis=1),
        "pres_sph": d["pres"][fluid],
        "rho_sph": d["rho"][fluid],
    }


def save_gif(frames: list[dict],
             solv: Solv,
             field: str = "pres",
             fps: int = 20,
             clip: float = 99.0,
             ani_dir: str = ANI_DIR,
             name: str = "dam_break") -> str:
    """
    Draw the collected frames as a gif.

    Fluid particles are coloured by the chosen field on a colour scale that is
    fixed over the whole run, so the colours mean the same thing in every
    frame. Boundary particles are drawn in grey.

    frames: [collect_frame(...), ...] in time order
    solv: property of the simulation, used for the axis limits
    field: "pres", "vel" or "rho"
    fps: frames per second of the gif
    clip: percentile the colour scale tops out at. Impact pressure spikes on a
        few particles are an order of magnitude above the bulk, and scaling to
        the absolute maximum would push the whole fluid into one colour
    ani_dir: directory the gif is written into
    name: file name without extension

    return: path of the written gif
    """
    if not frames:
        raise ValueError("no frames to write")
    key = {"vel": "vel_sph", "pres": "pres_sph", "rho": "rho_sph"}[field]
    label = {"vel": "|v| [m/s]", "pres": "pressure [Pa]", "rho": "density [kg/m^3]"}[field]

    os.makedirs(ani_dir, exist_ok=True)
    path = os.path.join(ani_dir, f"{name}.gif")

    # One colour scale for the whole run, topped out at the clip percentile
    every = np.concatenate([f[key] for f in frames])
    vmin = float(every.min())
    vmax = float(np.percentile(every, clip))
    if vmax <= vmin:
        vmax = vmin + 1.0
    extend = "max" if every.max() > vmax else "neither"

    pad = solv.bnd_layer * solv.dx
    dpi = 100
    fig_w = 8.0
    ax_frac = 0.78                              # axes width / figure width
    x_span = solv.tank_width + 2.0 * pad
    # Marker area in points^2, picked so a marker is dx wide and the particles
    # just touch at rest
    size = (solv.dx * fig_w * ax_frac * 72.0 / x_span) ** 2

    fig, ax = plt.subplots(figsize=(fig_w, fig_w * 0.5), dpi=dpi)
    ax.set_xlim(-pad, solv.tank_width + pad)
    ax.set_ylim(-pad, solv.tank_height + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    ax.scatter(frames[0]["pos_bnd"][:, 0], frames[0]["pos_bnd"][:, 1],
               s=size, c="0.6", marker="o", linewidths=0)
    sc = ax.scatter(frames[0]["pos_sph"][:, 0], frames[0]["pos_sph"][:, 1],
                    s=size, c=frames[0][key], cmap="viridis",
                    vmin=vmin, vmax=vmax, marker="o", linewidths=0)
    fig.colorbar(sc, ax=ax, label=label, fraction=0.03, pad=0.02, extend=extend)
    title = ax.set_title(f"t = {frames[0]['t']:.3f} s")
    fig.tight_layout()

    def draw(i: int):
        f = frames[i]
        sc.set_offsets(f["pos_sph"])
        sc.set_array(f[key])
        title.set_text(f"t = {f['t']:.3f} s")
        return sc, title

    ani = FuncAnimation(fig, draw, frames=len(frames), blit=False)
    ani.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path

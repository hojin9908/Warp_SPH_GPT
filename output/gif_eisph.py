"""EISPH rendering from loaded positions, without cavity generator settings."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from input.Config import Solv


def save_cavity_gif(positions: np.ndarray, velocity_frames: list[np.ndarray],
                    times: list[float], solv: Solv,
                    path: str | Path | None = None) -> Path:
    """Render fixed x-z points with physical speed [m/s] and velocity arrows.

    Bounds come from the input cloud. Point ordering, translated coordinates,
    and non-square point sets do not depend on generation settings.
    """
    if len(velocity_frames) != len(times) or not velocity_frames:
        raise ValueError("velocity_frames and times must have the same non-zero length")
    if positions.ndim != 2 or positions.shape[1] != 3 or not len(positions):
        raise ValueError("positions must be a nonempty [N,3] array")
    if not np.isfinite(positions).all():
        raise ValueError("positions must be finite")
    for velocity in velocity_frames:
        if velocity.shape != positions.shape or not np.isfinite(velocity).all():
            raise ValueError("every velocity frame must be finite and match positions")
    destination = Path(solv.animation_path if path is None else path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    speed = [np.linalg.norm(v[:, (0, 2)], axis=1) for v in velocity_frames]
    vmax = max(max(float(v.max()) for v in speed), 1e-8)
    x, z = positions[:, 0], positions[:, 2]
    span = max(float(np.ptp(x)), float(np.ptp(z)), solv.h)
    pad = 0.04 * span
    stride = max(1, len(positions) // 150)
    sample = slice(None, None, stride)
    fig, ax = plt.subplots(figsize=(6.1, 5.5))
    dots = ax.scatter(x, z, c=speed[0], cmap="turbo", vmin=0, vmax=vmax,
                      s=max(3.0, min(70.0, 25000.0 / len(positions))), marker="s")
    v0 = velocity_frames[0]
    arrows = ax.quiver(x[sample], z[sample], v0[sample, 0], v0[sample, 2],
                      color="white", angles="xy", scale_units="xy",
                      scale=15.0*vmax/span, width=0.003)
    ax.set(xlabel="x [m]", ylabel="z [m]", aspect="equal",
           xlim=(x.min()-pad, x.max()+pad), ylim=(z.min()-pad, z.max()+pad))
    title = ax.set_title(f"Eulerian ISPH   t = {times[0]:.3f} s")
    fig.colorbar(dots, ax=ax, pad=0.03).set_label("speed [m/s]")
    fig.tight_layout()

    def update(frame):
        dots.set_array(speed[frame])
        velocity = velocity_frames[frame]
        arrows.set_UVC(velocity[sample, 0], velocity[sample, 2])
        title.set_text(f"Eulerian ISPH   t = {times[frame]:.3f} s")
        return dots, arrows, title

    animation = FuncAnimation(fig, update, frames=len(times),
                              interval=1000.0/solv.gif_fps, blit=False)
    animation.save(destination, writer=PillowWriter(fps=solv.gif_fps), dpi=100)
    plt.close(fig)
    return destination.resolve()

from pathlib import Path

import matplotlib

matplotlib.use("Agg")               # headless rendering, no interactive window

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from input.Config import Solv


def save_cavity_gif(positions: np.ndarray,
                    velocity_frames: list[np.ndarray],
                    times: list[float],
                    solv: Solv,
                    path: str | Path | None = None) -> Path:
    """
    Render the EISPH lid-driven cavity as a two-dimensional GIF.

    positions: fixed host positions [N_sph,3], ordered on the x-z lattice
    velocity_frames: corrected host velocities [frame][N_sph,3]
    times: physical time of each stored frame [s]
    solv: cavity geometry, lid velocity and GIF configuration
    path: optional output path; solv.animation_path is used when omitted

    The normalized speed colour scale is fixed to [0,1] for all frames.
    White arrows show the x-z velocity components on a coarser display grid.

    return: absolute path of the saved GIF
    """
    # Reject inconsistent host data before creating an output file.
    if len(velocity_frames) != len(times) or not velocity_frames:
        raise ValueError("velocity_frames and times must have the same non-zero length")

    n = solv.cells
    expected_shape = (n * n, 3)
    if positions.shape != expected_shape:
        raise ValueError(f"expected positions with shape {expected_shape}")
    for velocity in velocity_frames:
        if velocity.shape != expected_shape or not np.all(np.isfinite(velocity)):
            raise ValueError("every velocity frame must be finite and match positions")

    destination = Path(solv.animation_path if path is None else path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stride = max(1, n // 12)
    x = positions[:, 0].reshape(n, n)
    z = positions[:, 2].reshape(n, n)

    def fields(frame: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Reshape one host frame to regular x-z fields.

        frame: stored frame index
        return: x velocity, z velocity and speed / lid_velocity [n,n]
        """
        velocity = velocity_frames[frame].reshape(n, n, 3)
        u = velocity[:, :, 0]
        w = velocity[:, :, 2]
        return u, w, np.sqrt(u * u + w * w) / solv.lid_velocity

    u0, w0, speed0 = fields(0)
    fig, ax = plt.subplots(figsize=(6.1, 5.5))
    # A fixed colour range preserves the meaning of colour through time.
    image = ax.imshow(speed0, origin="lower", extent=(0.0, solv.length, 0.0, solv.length),
                      cmap="turbo", vmin=0.0, vmax=1.0,
                      interpolation="bilinear")
    quiver = ax.quiver(x[::stride, ::stride], z[::stride, ::stride],
                       u0[::stride, ::stride], w0[::stride, ::stride],
                       color="white", angles="xy", scale_units="xy", scale=2.0,
                       width=0.004, pivot="mid")
    ax.plot([0, solv.length, solv.length, 0, 0],
            [0, 0, solv.length, solv.length, 0], color="black", linewidth=2.0)
    ax.annotate("moving lid", xy=(0.78 * solv.length, 1.035 * solv.length),
                xytext=(0.22 * solv.length, 1.035 * solv.length),
                arrowprops={"arrowstyle": "->", "lw": 2.0},
                ha="center", va="center")
    ax.set(xlabel="x / L", ylabel="z / L", aspect="equal",
           xlim=(-0.02 * solv.length, 1.02 * solv.length),
           ylim=(-0.02 * solv.length, 1.09 * solv.length))
    title = ax.set_title(f"Eulerian ISPH lid-driven cavity   t = {times[0]:.3f} s")
    colorbar = fig.colorbar(image, ax=ax, pad=0.03)
    colorbar.set_label(r"$|u| / U_{lid}$")
    fig.tight_layout()

    def update(frame: int):
        """
        Update the colour field, velocity arrows and physical-time title.

        frame: stored frame index
        return: Matplotlib artists for FuncAnimation; blit=False redraws the axes
        """
        u, w, speed = fields(frame)
        image.set_data(speed)
        quiver.set_UVC(u[::stride, ::stride], w[::stride, ::stride])
        title.set_text(f"Eulerian ISPH lid-driven cavity   t = {times[frame]:.3f} s")
        return image, quiver, title

    # PillowWriter creates the only user-visible EISPH result.
    animation = FuncAnimation(fig, update, frames=len(velocity_frames),
                              interval=1000.0 / solv.gif_fps, blit=False)
    animation.save(destination, writer=PillowWriter(fps=solv.gif_fps), dpi=100)
    plt.close(fig)
    return destination.resolve()

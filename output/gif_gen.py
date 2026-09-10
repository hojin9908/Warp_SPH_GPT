import os

import matplotlib
matplotlib.use("Agg")               # headless rendering, no interactive window

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from input.Config import Solv
from output.output import SPH_TYPE, gather_state
from input.struct import SPHptl, BNDptl, DEMptl, DEMBNDptl

ANI_DIR = "animation/3d"


def collect_frame(P_sph: SPHptl,
                  P_bnd: BNDptl,
                  t: float,
                  P_dem: DEMptl = None,
                  P_dem_bnd: DEMBNDptl = None) -> dict:
    """
    Copy a synchronized 3D state from the device for later GIF rendering.

    P_sph, P_bnd: fluid / dummy boundary particle structures
    t: physical time of the stored state [s]
    P_dem, P_dem_bnd: optional moving / fixed DEM sphere structures

    return: host arrays with all three position components [N,3], radii [m],
        and fluid velocity magnitude, pressure and density. Indices stay unchanged.
    """
    d = gather_state(P_sph, P_bnd)
    fluid = d["type"] == SPH_TYPE
    frame = {
        "t": t,
        "pos_sph": d["pos"][fluid],
        "pos_bnd": d["pos"][~fluid],
        "vel_sph": np.linalg.norm(d["vel"][fluid], axis=1),
        "pres_sph": d["pres"][fluid],
        "rho_sph": d["rho"][fluid],
    }
    # DEM geometry is in meters, sharing the SPH coordinate system.
    if P_dem is not None:
        frame["pos_dem"] = P_dem.pos.numpy()
        frame["radius_dem"] = P_dem.radius.numpy()
    if P_dem_bnd is not None:
        frame["pos_dem_bnd"] = P_dem_bnd.pos.numpy()
        frame["radius_dem_bnd"] = P_dem_bnd.radius.numpy()
    return frame


def sphere_faces(n_lat: int = 10, n_lon: int = 16) -> np.ndarray:
    """
    Build quadrilateral faces of a unit sphere centered at the origin.

    n_lat, n_lon: latitude / longitude subdivisions for display only
    return: unit-coordinate vertices [n_lat*n_lon,4,3]; y is the polar axis
    The caller scales by each physical radius and translates to each center.
    """
    latitude, longitude = np.meshgrid(np.linspace(0.0, np.pi, n_lat + 1),
                                     np.linspace(0.0, 2.0*np.pi, n_lon + 1), indexing="ij")
    points = np.stack([np.sin(latitude)*np.cos(longitude), np.cos(latitude),
                       np.sin(latitude)*np.sin(longitude)], axis=-1)
    return np.stack([points[:-1, :-1], points[1:, :-1],
                     points[1:, 1:], points[:-1, 1:]], axis=2).reshape(-1, 4, 3)


def save_gif(frames: list[dict],
             solv: Solv,
             field: str = "pres",
             fps: int = 20,
             clip: float = 99.0,
             ani_dir: str = ANI_DIR,
             name: str = "dam_break_3d") -> str:
    """
    Render SPH points and physical DEM sphere surfaces together in a 3D GIF.

    frames: synchronized host states from collect_frame, in time order
    solv: tank dimensions and SPH spacing; y is vertical, z is tank depth
    field: fluid colour field (pres, vel or rho), with a fixed scale for all frames
    fps: playback frames per second; physical time is printed on every frame
    clip: percentile at which the fluid colour scale saturates
    ani_dir, name: output directory and GIF basename

    return: saved GIF path
    All moving particles are drawn. Fixed particle walls use a tank outline
    to keep the interior visible; VTK retains every wall particle.
    """
    if not frames:
        raise ValueError("no frames to write")
    key = {"vel": "vel_sph", "pres": "pres_sph", "rho": "rho_sph"}[field]
    label = {"vel": "|v| [m/s]", "pres": "pressure [Pa]", "rho": "density [kg/m^3]"}[field]
    os.makedirs(ani_dir, exist_ok=True)
    path = os.path.join(ani_dir, f"{name}.gif")

    # One colour scale preserves the meaning of colour throughout the animation.
    every = np.concatenate([f[key] for f in frames])
    vmin = float(every.min())
    vmax = max(float(np.percentile(every, clip)), vmin + 1.0)
    extend = "max" if every.max() > vmax else "neither"
    pad = solv.bnd_layer * solv.dx
    lower = np.array([-pad, -pad, -pad])
    upper = np.array([solv.tank_width + pad, solv.tank_height + pad, solv.tank_depth + pad])
    # Include all moving particles, even above the open top, with fixed axes in time.
    for frame in frames:
        lower = np.minimum(lower, frame["pos_sph"].min(axis=0) - solv.dx)
        upper = np.maximum(upper, frame["pos_sph"].max(axis=0) + solv.dx)
        if "pos_dem" in frame:
            lower = np.minimum(lower, (frame["pos_dem"] - frame["radius_dem"][:, None]).min(axis=0) - pad)
            upper = np.maximum(upper, (frame["pos_dem"] + frame["radius_dem"][:, None]).max(axis=0) + pad)

    fig = plt.figure(figsize=(10, 6.4), dpi=100)
    # Keep DEM visible through the translucent fluid, as an interior-particle overlay.
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    ax.set_xlim(lower[0], upper[0])
    # Plot (x,z,y) so physical y is vertical without changing stored coordinates.
    ax.set_ylim(lower[2], upper[2])
    ax.set_zlim(lower[1], upper[1])
    ax.set_box_aspect((upper - lower)[[0, 2, 1]])  # the same meter scale on each axis
    ax.view_init(elev=24, azim=-65)
    ax.set_proj_type("ortho")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m] (depth)")
    ax.set_zlabel("y [m] (height)")
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
    # Twelve outline edges indicate the tank; its upper opening has no wall surface.
    corners = np.array([[x, y, z] for x in (0.0, solv.tank_width)
                        for y in (0.0, solv.tank_height) for z in (0.0, solv.tank_depth)])
    for i in range(len(corners)):
        for j in range(i + 1, len(corners)):
            if np.count_nonzero(corners[i] != corners[j]) == 1:
                edge = corners[[i, j]]
                ax.plot(edge[:, 0], edge[:, 2], edge[:, 1], color="0.55", linewidth=0.7, alpha=0.65, zorder=1)

    # Translucent SPH points let the submerged DEM spheres remain visible.
    pos = frames[0]["pos_sph"]
    sc = ax.scatter(pos[:, 0], pos[:, 2], pos[:, 1], c=frames[0][key],
                    s=5, cmap="viridis", vmin=vmin, vmax=vmax,
                    alpha=0.30, depthshade=False, linewidths=0, zorder=2)
    dem = None
    if "pos_dem" in frames[0]:
        unit_faces = sphere_faces()
        # Fixed lighting displays the sphere curvature without changing its radius.
        normals = unit_faces.mean(axis=1)
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        light = np.array([-0.5, 1.0, -0.8])
        light /= np.linalg.norm(light)
        shade = 0.45 + 0.55 * np.maximum(normals @ light, 0.0)
        colours = shade[:, None] * np.array([1.0, 0.55, 0.05])
        dem_pos = frames[0]["pos_dem"]
        radii = frames[0]["radius_dem"]
        vertices = dem_pos[:, None, None, :] + radii[:, None, None, None] * unit_faces[None]
        dem = Poly3DCollection(vertices.reshape(-1, 4, 3)[:, :, [0, 2, 1]],
                               facecolors=np.tile(colours, (len(dem_pos), 1)),
                               edgecolors="none", zsort="average", zorder=3)
        ax.add_collection3d(dem)
    handles = [Line2D([], [], marker="o", linestyle="none", color="teal", label="SPH fluid")]
    if dem is not None:
        handles.append(Line2D([], [], marker="o", linestyle="none", color="darkorange", label="DEM spheres"))
    ax.legend(handles=handles, loc="upper left", framealpha=0.85)
    fig.colorbar(sc, ax=ax, label=label, fraction=0.025, pad=0.06, shrink=0.68, extend=extend)
    title = ax.set_title(f"3D Dam Break | t = {frames[0]['t']:.4f} s")
    fig.subplots_adjust(left=0.02, right=0.88, bottom=0.06, top=0.92)

    def draw(i: int):
        """
        Update all three SPH coordinates and DEM sphere surfaces for frame i.

        i: host-frame index, independent of particle indices
        return: artists for FuncAnimation; blit=False redraws the 3D scene
        """
        frame = frames[i]
        pos = frame["pos_sph"]
        sc._offsets3d = (pos[:, 0], pos[:, 2], pos[:, 1])
        sc.set_array(frame[key])
        if dem is not None:
            vertices = (frame["pos_dem"][:, None, None, :]
                        + frame["radius_dem"][:, None, None, None] * unit_faces[None])
            dem.set_verts(vertices.reshape(-1, 4, 3)[:, :, [0, 2, 1]])
        title.set_text(f"3D Dam Break | t = {frame['t']:.4f} s")
        return sc, title

    ani = FuncAnimation(fig, draw, frames=len(frames), blit=False)
    ani.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path

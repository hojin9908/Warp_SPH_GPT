import os

import numpy as np

from input.struct import SPHptl, BNDptl, DEMptl, DEMBNDptl

OUT_DIR = "result/3d"

SPH_TYPE = 1        # fluid particle
BND_TYPE = 0        # dummy boundary particle
DEM_TYPE = 2        # moving DEM particle
DEM_BND_TYPE = 3    # fixed DEM boundary particle


def _fmt_float(a: np.ndarray, per_line: int = 6) -> str:
    """Flatten to ascii, per_line values on each line."""
    v = ["%.6g" % x for x in np.asarray(a, dtype=np.float64).ravel()]
    return "\n".join(" ".join(v[i:i + per_line]) for i in range(0, len(v), per_line))


def _fmt_int(a: np.ndarray, per_line: int = 20) -> str:
    """Flatten integer connectivity / type arrays to ASCII with per_line values per row."""
    v = ["%d" % x for x in np.asarray(a).ravel()]
    return "\n".join(" ".join(v[i:i + per_line]) for i in range(0, len(v), per_line))


def gather_state(P_sph: SPHptl,
                 P_bnd: BNDptl) -> dict[str, np.ndarray]:
    """
    Copy SPH and BND particles from the device into one set of numpy arrays.

    The two structures are stacked so that ParaView shows the fluid and the
    tank in a single dataset. Use a Threshold filter on "type" to split them
    again (1 = fluid, 0 = boundary).

    P_sph: Particle structure of SPH particles      [N_sph]
    P_bnd: Particle structure of BND particles      [N_bnd]

    return: dict of [N_sph + N_bnd] arrays
    """
    n_sph = P_sph.pos.shape[0]
    n_bnd = P_bnd.pos.shape[0]
    return {
        "pos":  np.vstack([P_sph.pos.numpy(),  P_bnd.pos.numpy()]),
        "vel":  np.vstack([P_sph.vel.numpy(),  P_bnd.vel.numpy()]),
        "rho":  np.concatenate([P_sph.rho.numpy(),  P_bnd.rho.numpy()]),
        "pres": np.concatenate([P_sph.pres.numpy(), P_bnd.pres.numpy()]),
        "m":    np.concatenate([P_sph.m.numpy(), P_bnd.m.numpy()]),
        # Dummy particles have no fluid coupling fields; placeholders keep one point order.
        "porosity": np.concatenate([P_sph.porosity.numpy(), np.ones(n_bnd)]),
        "pgf": np.vstack([P_sph.pgf.numpy(), np.zeros((n_bnd, 3))]),
        "acc_dem": np.vstack([P_sph.acc_dem.numpy(), np.zeros((n_bnd, 3))]),
        "type": np.concatenate([np.full(n_sph, SPH_TYPE, dtype=np.int32),
                                np.full(n_bnd, BND_TYPE, dtype=np.int32)]),
    }


def save_vtk(P_sph: SPHptl,
             P_bnd: BNDptl,
             step: int,
             t: float,
             out_dir: str = OUT_DIR,
             name: str = "ptl") -> tuple[str, float]:
    """
    Write one frame of the particle state as a VTK XML PolyData file.

    Every particle becomes a vertex cell, so ParaView renders the file as a
    point cloud straight away. Position, velocity, density, pressure and the
    particle type are written as point data.

    P_sph: Particle structure of SPH particles      [N_sph]
    P_bnd: Particle structure of BND particles      [N_bnd]
    step: current step of the simulation
    t: physical time of this frame [s]
    out_dir: directory the frame is written into
    name: file name prefix

    return: (file name, t) -- feed the collected list to save_pvd
    """
    d = gather_state(P_sph, P_bnd)
    return _save_vtp(d, step, t, out_dir, name, "pres")


def save_dem_vtk(P_dem: DEMptl,
                 P_dem_bnd: DEMBNDptl,
                 step: int,
                 t: float,
                 out_dir: str = OUT_DIR,
                 name: str = "dem") -> tuple[str, float]:
    """
    Write moving DEM and kinematic fixed-wall geometry in one dataset.

    radius can be used as the ParaView Sphere Glyph scale; type=2 selects
    moving beads and type=3 selects the fixed wall. Moving-only vector fields
    use zero placeholders on fixed-wall points.

    P_dem: Particle structure of moving DEM spheres [N_dem]
    P_dem_bnd: Particle structure of fixed DEM spheres [N_dem_bnd]
    step: number of completed integration steps
    t: physical time of the frame [s]
    out_dir: directory for the VTP file
    name: file name prefix, kept distinct from the SPH prefix

    return: (file name, t) -- feed the collected list to save_pvd
    """
    n_dem, n_bnd = P_dem.pos.shape[0], P_dem_bnd.pos.shape[0]
    d = {}
    # Stack fields that are physically stored by both moving and fixed spheres.
    for key in ("pos", "vel", "omega", "radius", "rho", "m", "volume", "inertia"):
        d[key] = np.concatenate([getattr(P_dem, key).numpy(), getattr(P_dem_bnd, key).numpy()])
    # The kinematic wall has no acceleration, force or torque state.
    for key in ("acc", "force", "torque"):
        d[key] = np.vstack([getattr(P_dem, key).numpy(), np.zeros((n_bnd, 3))])
    d["type"] = np.concatenate([np.full(n_dem, DEM_TYPE, dtype=np.int32),
                                 np.full(n_bnd, DEM_BND_TYPE, dtype=np.int32)])
    # Only moving spheres participate in fluid coupling; fixed-wall values are placeholders.
    for key in ("drag", "pressure_force"):
        d[key] = np.vstack([getattr(P_dem, key).numpy(), np.zeros((n_bnd, 3))])
    d["porosity"] = np.concatenate([P_dem.porosity.numpy(), np.ones(n_bnd)])
    return _save_vtp(d, step, t, out_dir, name, "radius")


def _save_vtp(d: dict[str, np.ndarray], step: int, t: float,
              out_dir: str, name: str, scalar: str) -> tuple[str, float]:
    """
    Write one particle state as VTK XML PolyData with one vertex per particle.

    d: host arrays with pos [N,3], scalar fields [N], vector fields [N,3],
        and integer particle type [N]; all arrays share the same point order
    step: number used in the zero-padded file name
    t: physical time forwarded to the PVD entry [s]
    out_dir: directory for the VTP file
    name: phase-specific file name prefix
    scalar: default scalar field displayed by the VTK reader (pres or radius)

    return: (file name, t), using a file name relative to out_dir
    """
    os.makedirs(out_dir, exist_ok=True)
    n = d["pos"].shape[0]
    fname = f"{name}_{step:06d}.vtp"

    with open(os.path.join(out_dir, fname), "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n')
        f.write('  <PolyData>\n')
        f.write(f'    <Piece NumberOfPoints="{n}" NumberOfVerts="{n}" '
                'NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="0">\n')
        # Geometry: raw xyz particle centers with z vertical and y depth.
        f.write('      <Points>\n')
        f.write('        <DataArray type="Float32" Name="Points" '
                'NumberOfComponents="3" format="ascii">\n')
        f.write(_fmt_float(d["pos"]) + "\n")
        f.write('        </DataArray>\n')
        f.write('      </Points>\n')
        # One vertex cell per particle
        f.write('      <Verts>\n')
        f.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
        f.write(_fmt_int(np.arange(n)) + "\n")
        f.write('        </DataArray>\n')
        f.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
        f.write(_fmt_int(np.arange(1, n + 1)) + "\n")
        f.write('        </DataArray>\n')
        f.write('      </Verts>\n')
        # State fields: vectors have three components; particle type remains integer.
        f.write(f'      <PointData Scalars="{scalar}" Vectors="vel">\n')
        for key, values in d.items():
            if key == "pos":
                continue
            dtype = "Int32" if key == "type" else "Float32"
            components = ' NumberOfComponents="3"' if values.ndim == 2 else ""
            f.write(f'        <DataArray type="{dtype}" Name="{key}"{components} format="ascii">\n')
            f.write((_fmt_int(values) if key == "type" else _fmt_float(values)) + "\n")
            f.write('        </DataArray>\n')
        f.write('      </PointData>\n')
        f.write('    </Piece>\n')
        f.write('  </PolyData>\n')
        f.write('</VTKFile>\n')

    return fname, t


def save_pvd(frames: list[tuple[str, float]],
             out_dir: str = OUT_DIR,
             name: str = "ptl") -> str:
    """
    Write the ParaView collection file that ties the frames into a time series.

    Open this one file in ParaView and the animation carries the physical time
    of each frame, not just the frame index.

    frames: [(file name, t), ...] as returned by save_vtk
    out_dir: directory the collection is written into
    name: file name prefix

    return: path of the written .pvd file
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.pvd")
    with open(path, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        f.write('  <Collection>\n')
        for fname, t in frames:
            f.write(f'    <DataSet timestep="{t:.9g}" group="" part="0" file="{fname}"/>\n')
        f.write('  </Collection>\n')
        f.write('</VTKFile>\n')
    return path

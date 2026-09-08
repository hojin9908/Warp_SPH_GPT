import os

import numpy as np

from input.struct import SPHptl, BNDptl

OUT_DIR = "result"

SPH_TYPE = 1        # fluid particle
BND_TYPE = 0        # dummy boundary particle


def _fmt_float(a: np.ndarray, per_line: int = 6) -> str:
    """Flatten to ascii, per_line values on each line."""
    v = ["%.6g" % x for x in np.asarray(a, dtype=np.float64).ravel()]
    return "\n".join(" ".join(v[i:i + per_line]) for i in range(0, len(v), per_line))


def _fmt_int(a: np.ndarray, per_line: int = 20) -> str:
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
    os.makedirs(out_dir, exist_ok=True)
    d = gather_state(P_sph, P_bnd)
    n = d["pos"].shape[0]
    fname = f"{name}_{step:06d}.vtp"

    with open(os.path.join(out_dir, fname), "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n')
        f.write('  <PolyData>\n')
        f.write(f'    <Piece NumberOfPoints="{n}" NumberOfVerts="{n}" '
                'NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="0">\n')
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
        f.write('      <PointData Scalars="pres" Vectors="vel">\n')
        for key in ("rho", "pres"):
            f.write(f'        <DataArray type="Float32" Name="{key}" format="ascii">\n')
            f.write(_fmt_float(d[key]) + "\n")
            f.write('        </DataArray>\n')
        f.write('        <DataArray type="Float32" Name="vel" '
                'NumberOfComponents="3" format="ascii">\n')
        f.write(_fmt_float(d["vel"]) + "\n")
        f.write('        </DataArray>\n')
        f.write('        <DataArray type="Int32" Name="type" format="ascii">\n')
        f.write(_fmt_int(d["type"]) + "\n")
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

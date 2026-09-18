"""SOPHIA-style numeric-header particle input (project-specific column IDs).

Each file has a whitespace-separated integer header and one particle per row.
IDs restart at 1 for each kind, following the Notion field order and expanding
vec3 as x/y/z. Computed fields have no IDs. Column order may be permuted.
This is an initial-condition format, not a restart/checkpoint format.
"""
from pathlib import Path

import numpy as np
import warp as wp

from input.struct import SPHptl, BNDptl, DEMptl, DEMBNDptl
from input.struct_eisph import EISPHptl


# (field, scalar-column count), in the documented Notion order.
INPUT_FIELDS = {
    "SPH": (("pos", 3), ("vel", 3), ("rho", 1), ("m", 1)),
    "BND": (("pos", 3), ("vel", 3), ("rho", 1), ("m", 1)),
    "DEM": (("pos", 3), ("vel", 3), ("omega", 3), ("radius", 1),
            ("rho", 1), ("m", 1), ("inertia", 1), ("K", 1),
            ("eta", 1), ("mu", 1), ("h", 1)),
    "DEMBND": (("pos", 3), ("vel", 3), ("omega", 3), ("radius", 1),
               ("rho", 1), ("m", 1), ("inertia", 1), ("K", 1),
               ("eta", 1), ("mu", 1)),
    "EISPH": (("pos", 3), ("vel", 3), ("rho", 1), ("m", 1)),
    "EISPHBND": (("pos", 3), ("vel_bc", 3), ("rho", 1), ("m", 1),
                 ("mirror", 1)),
}
STRUCTURES = {"SPH": SPHptl, "BND": BNDptl, "DEM": DEMptl,
              "DEMBND": DEMBNDptl, "EISPH": EISPHptl, "EISPHBND": EISPHptl}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def column_indices(kind: str) -> dict[str, tuple[int, ...]]:
    """Return the documented numeric IDs for each input field."""
    if kind not in INPUT_FIELDS:
        raise ValueError(f"unknown particle kind {kind!r}; choose {tuple(INPUT_FIELDS)}")
    index = 1
    result = {}
    for field, width in INPUT_FIELDS[kind]:
        result[field] = tuple(range(index, index + width))
        index += width
    return result


def _read_table(path: Path, kind: str) -> dict[str, np.ndarray]:
    columns = column_indices(kind)
    required = {i for ids in columns.values() for i in ids}
    with path.open(encoding="utf-8-sig") as stream:
        try:
            header = [int(token) for token in stream.readline().split()]
        except ValueError as exc:
            raise ValueError(f"{path}:1: header must contain integer column IDs") from exc
        if not header or len(header) != len(set(header)):
            raise ValueError(f"{path}:1: empty header or duplicate column IDs")
        if set(header) != required:
            raise ValueError(f"{path}:1: missing IDs {sorted(required-set(header))}; "
                             f"unknown IDs {sorted(set(header)-required)}")
        rows = []
        for line_number, line in enumerate(stream, 2):
            tokens = line.split()
            if len(tokens) != len(header):
                raise ValueError(f"{path}:{line_number}: expected {len(header)} values, "
                                 f"got {len(tokens)}")
            try:
                row = [float(token) for token in tokens]
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid numeric value") from exc
            if not np.isfinite(row).all() or np.any(np.abs(row) > np.finfo(np.float32).max):
                raise ValueError(f"{path}:{line_number}: values must be finite float32 numbers")
            rows.append(row)
    data = np.asarray(rows, dtype=np.float64).reshape(-1, len(header))
    result = {}
    for field, ids in columns.items():
        values = data[:, [header.index(i) for i in ids]]
        result[field] = values[:, 0] if len(ids) == 1 else values
    for field in ("rho", "m", "radius", "inertia", "K", "h"):
        if field in result and np.any(result[field].astype(np.float32) <= 0.0):
            raise ValueError(f"{path}: {field} must be positive in float32")
    for field in ("eta", "mu"):
        if field in result and np.any(result[field] < 0.0):
            raise ValueError(f"{path}: {field} must be nonnegative")
    if "mirror" in result:
        ids = result["mirror"]
        if np.any((ids != np.floor(ids)) | (ids < 0) | (ids > np.iinfo(np.int32).max)):
            raise ValueError(f"{path}: mirror must be a nonnegative int32 fluid row index")
    if "radius" in result:
        volume = (4.0 / 3.0) * np.pi * result["radius"] ** 3
        if not np.allclose(result["m"], result["rho"] * volume, rtol=1e-5, atol=0.0):
            raise ValueError(f"{path}: sphere mass must equal rho*(4/3)*pi*radius^3")
        if np.any(volume.astype(np.float32) <= 0) or not np.isfinite(volume.astype(np.float32)).all():
            raise ValueError(f"{path}: derived sphere volume is outside float32 range")
    return result


def input_parser(file_path: str | Path, kind: str | None = None,
                 device: str = "cpu"):
    """Read input_<kind>.txt and return a fully allocated Warp particle struct.

    kind may be SPH, BND, DEM, DEMBND, EISPH or EISPHBND. If omitted, it
    is inferred from the filename. Every documented input column is required.
    Header-only files produce empty structs; the solver validates required sets.
    Ghost mirror indices are zero-based rows in input_EISPH.txt. Pair-dependent
    validation and initial ghost velocity are handled by load_eisph_particles.
    """
    path = Path(file_path)
    kind = path.stem.removeprefix("input_") if kind is None else kind
    data = _read_table(path, kind)
    particle = STRUCTURES[kind]()
    n = len(data["pos"])
    # Initialize every field, including empty old/new contact CSR buffers.
    for name, var in particle._cls.vars.items():
        size = n
        if name.startswith("contact_"):
            size = n + 1 if "offset" in name else 0
        elif name.startswith("tang_"):
            size = 0
        setattr(particle, name, wp.zeros(size, dtype=var.type.dtype, device=device))
    for name, values in data.items():
        dtype = particle._cls.vars[name].type.dtype
        setattr(particle, name, wp.array(values, dtype=dtype, device=device))
    if kind in ("SPH", "BND"):
        wp.copy(particle.rho_raw, particle.rho)
    if kind in ("SPH", "DEM"):
        particle.porosity.fill_(1.0)
    if kind in ("DEM", "DEMBND"):
        volume = (4.0 / 3.0) * np.pi * data["radius"] ** 3
        particle.volume = wp.array(volume, dtype=float, device=device)
    if kind == "EISPH":
        wp.copy(particle.vel_star, particle.vel)
        particle.mirror = wp.array(np.arange(n), dtype=wp.int32, device=device)
    return particle


def input_directory(solv) -> Path:
    directory = Path(solv.input_dir)
    return directory if directory.is_absolute() else PROJECT_ROOT / directory


def validate_dem_particles(solv, dem, boundary) -> tuple[float, float, float, float]:
    """Validate contact dt from loaded materials and return radius/h maxima.

    The conservative reduced mass min(m)/2 bounds every moving pair. Contact
    laws currently use the moving subject's K/eta/mu (boundary values are
    stored metadata). Geometry/material fields stay fixed during a run.
    """
    solv.validate_dem()
    if dem.pos.shape[0] == 0:
        raise ValueError("DEM enabled but input_DEM.txt contains no particles")
    mass = dem.m.numpy().astype(float)
    stiffness = dem.K.numpy().astype(float)
    damping = dem.eta.numpy().astype(float)
    m_eff = 0.5 * mass.min()
    limit = solv.dem_dt_safety * min(np.sqrt(m_eff / stiffness.max()),
                                    m_eff / max(float(damping.max()), 1e-20))
    if solv.dt > limit:
        raise ValueError(f"dt exceeds loaded DEM contact limit {limit:.6g} s")
    return dem_search_scales(dem, boundary)


def dem_search_scales(dem, boundary) -> tuple[float, float, float, float]:
    """Bounds for immutable loaded material arrays; cache once per run."""
    h = dem.h.numpy()
    return (float(dem.radius.numpy().max()),
            float(boundary.radius.numpy().max()) if boundary.pos.shape[0] else 0.0,
            float(h.max()), float(h[0]) if np.all(h == h[0]) else 0.0)


def load_sph_particles(solv):
    """Read the two or four WCSPH particle files selected by dem_enable."""
    solv.validate_sph()
    directory = input_directory(solv)
    sph, bnd = (input_parser(directory / f"input_{kind}.txt", kind, solv.device)
                for kind in ("SPH", "BND"))
    if sph.pos.shape[0] == 0:
        raise ValueError("input_SPH.txt must contain fluid particles")
    dem = dem_bnd = None
    if solv.dem_enable:
        dem, dem_bnd = (input_parser(directory / f"input_{kind}.txt", kind, solv.device)
                        for kind in ("DEM", "DEMBND"))
        validate_dem_particles(solv, dem, dem_bnd)
    return sph, bnd, dem, dem_bnd


def validate_eisph_particles(solv, fluid, boundary) -> None:
    """Validate a fixed planar EISPH cloud and its prescribed mirror mapping."""
    solv.validate_eisph()
    n = fluid.pos.shape[0]
    if n == 0 or boundary.pos.shape[0] == 0:
        raise ValueError("EISPH needs nonempty fluid and ghost files")
    if np.any(boundary.mirror.numpy() >= n):
        raise ValueError("EISPHBND mirror is outside the input_EISPH fluid row range")
    for particle in (fluid, boundary):
        if not np.allclose(particle.rho.numpy(), solv.rho0, rtol=1e-6, atol=0):
            raise ValueError("EISPH currently requires uniform rho equal to rho0")
        if not np.allclose(particle.pos.numpy()[:, 1], 0.0, atol=1e-8):
            raise ValueError("EISPH points must lie in the x-z plane (y=0)")
    # The bundled EISPH case has area mass m=rho*dx^2. Use the minimum
    # equivalent cell length for conservative initial predictor dt checks.
    dx = float(np.sqrt(fluid.m.numpy() / fluid.rho.numpy()).min())
    speed = max(float(np.linalg.norm(fluid.vel.numpy(), axis=1).max()),
                float(np.linalg.norm(boundary.vel_bc.numpy(), axis=1).max()))
    if speed > 0 and solv.dt > 0.25 * dx / speed:
        raise ValueError("dt exceeds the loaded EISPH initial advection limit")
    if solv.nu > 0 and solv.dt > 0.125 * dx * dx / solv.nu:
        raise ValueError("dt exceeds the loaded EISPH viscosity limit")


def load_eisph_particles(solv):
    """Read EISPH fluid/ghost files, validate IDs and initialize ghost states."""
    directory = input_directory(solv)
    fluid, boundary = (input_parser(directory / f"input_{kind}.txt", kind, solv.device)
                       for kind in ("EISPH", "EISPHBND"))
    validate_eisph_particles(solv, fluid, boundary)
    velocity = 2.0 * boundary.vel_bc.numpy() - fluid.vel.numpy()[boundary.mirror.numpy()]
    boundary.vel = wp.array(velocity, dtype=wp.vec3, device=solv.device)
    wp.copy(boundary.vel_star, boundary.vel)
    return fluid, boundary

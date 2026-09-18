"""Generate the six documented input files from the original example cases.

Run from the project root: python -m input_gen.generate
Edit input_gen/config.py for geometry/initial material values, then regenerate.
Runtime entry points never generate or overwrite particle input files.
"""
import argparse
from pathlib import Path

import numpy as np
import warp as wp

from input.input_reader import INPUT_FIELDS, PROJECT_ROOT, column_indices
from input_gen.config import GenerationConfig, CoupledGenerationConfig
from input_gen.gen_ptl import DamPtlGeneration
from input_gen.gen_dem import DEMPtlGeneration
from input_gen.gen_eisph import CavityPtlGeneration


def write_particle_file(path: str | Path, kind: str, particle) -> Path:
    """Write only input fields, with the numeric IDs used by input_parser."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    n = particle.pos.shape[0]
    columns = [getattr(particle, name).numpy().reshape(n, width)
               for name, width in INPUT_FIELDS[kind]]
    header = "\t".join(str(i) for ids in column_indices(kind).values() for i in ids)
    # Nine significant digits round-trip all float32 particle values.
    np.savetxt(destination, np.concatenate(columns, axis=1), fmt="%.9g",
               delimiter="\t", header=header, comments="", encoding="utf-8")
    return destination


def generate_files(output_dir: str | Path = PROJECT_ROOT / "input_file",
                   coupled: CoupledGenerationConfig | None = None,
                   cavity: GenerationConfig | None = None) -> dict[str, Path]:
    """Build the unchanged 3D dam-break and 2D cavity initial configurations."""
    coupled = CoupledGenerationConfig(device="cpu") if coupled is None else coupled
    cavity = GenerationConfig(device="cpu") if cavity is None else cavity
    wp.init()
    particles = dict(zip(("SPH", "BND"), DamPtlGeneration(coupled).build()))
    particles.update(zip(("DEM", "DEMBND"), DEMPtlGeneration(coupled).build()))
    particles.update(zip(("EISPH", "EISPHBND"), CavityPtlGeneration(cavity).build()))
    return {kind: write_particle_file(Path(output_dir) / f"input_{kind}.txt", kind, particle)
            for kind, particle in particles.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SPH/DEM/EISPH particle inputs")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "input_file")
    args = parser.parse_args()
    for kind, path in generate_files(args.output_dir).items():
        print(f"[{kind}] {path}")


if __name__ == "__main__":
    main()

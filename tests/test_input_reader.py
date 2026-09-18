"""File-format, initialization, and loaded-particle solver regression tests."""
from dataclasses import fields
from pathlib import Path
import tempfile
import unittest

import numpy as np
import warp as wp

from input.Config import Solv
from input.Config_SPH_DEM import Solv as CoupledSolv
from input.input_reader import (INPUT_FIELDS, column_indices, input_parser,
                                load_sph_particles, load_eisph_particles,
                                validate_dem_particles, dem_search_scales)
from input_gen.config import GenerationConfig, CoupledGenerationConfig
from input_gen.generate import generate_files, write_particle_file
from input_gen.gen_ptl import DamPtlGeneration
from input_gen.gen_dem import DEMPtlGeneration
from input_gen.gen_eisph import CavityPtlGeneration
from source.Simulation import SPHDEM_OneStep
from source.EISPH import run_eisph
from kernel.KERNEL_SPHDEM_interaction import (Kernel_prep_sphdem,
                                             Kernel_interaction_dem,
                                             Kernel_interaction_sph)


class InputReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.devices = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
        cls.temporary = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temporary.name)
        cls.generation = CoupledGenerationConfig(
            device="cpu", dx=0.04, h=0.052, tank_width=0.8, tank_height=0.8,
            tank_depth=0.32, fluid_width=0.4, fluid_height=0.4, fluid_depth=0.32,
            dem_nx=2, dem_ny=1, dem_nz=1, dem_origin_y=0.16, dem_origin_z=0.2)
        cls.cavity = GenerationConfig(device="cpu", dx=0.2, n_steps=10, output_step=5)
        cls.paths = generate_files(cls.directory, cls.generation, cls.cavity)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def runtime(self, device="cpu", eisph=False):
        generator = self.cavity if eisph else self.generation
        config = Solv if eisph else CoupledSolv
        values = {f.name: getattr(generator, f.name) for f in fields(config)
                  if hasattr(generator, f.name)}
        values.update(device=device, input_dir=str(self.directory))
        return config(**values)

    def test_documented_column_ids_and_solver_only_config(self):
        self.assertEqual(column_indices("SPH"),
                         {"pos": (1, 2, 3), "vel": (4, 5, 6), "rho": (7,), "m": (8,)})
        self.assertEqual(column_indices("DEM")["K"], (14,))
        self.assertEqual(column_indices("DEM")["h"], (17,))
        self.assertEqual(column_indices("EISPHBND")["mirror"], (9,))
        banned = {"dx", "bnd_layer", "dem_K", "dem_eta", "dem_mu", "dem_h",
                  "dem_radius", "dem_mass", "tank_width", "lid_velocity", "reynolds"}
        for config in (Solv, CoupledSolv):
            self.assertFalse(banned.intersection(f.name for f in fields(config)))
        for kind in INPUT_FIELDS:
            self.assertFalse({"pres", "acc", "volume", "flt", "porosity"}
                             .intersection(column_indices(kind)))

    def test_all_six_files_match_generated_structs_on_cpu_and_cuda(self):
        generated = dict(zip(("SPH", "BND"), DamPtlGeneration(self.generation).build()))
        generated.update(zip(("DEM", "DEMBND"), DEMPtlGeneration(self.generation).build()))
        generated.update(zip(("EISPH", "EISPHBND"), CavityPtlGeneration(self.cavity).build()))
        for device in self.devices:
            with self.subTest(device=device):
                loaded = dict(zip(("SPH", "BND", "DEM", "DEMBND"),
                                  load_sph_particles(self.runtime(device))))
                loaded.update(zip(("EISPH", "EISPHBND"),
                                  load_eisph_particles(self.runtime(device, True))))
                for kind, particle in loaded.items():
                    self.assertEqual(str(particle.pos.device), device)
                    for name in particle._cls.vars:
                        np.testing.assert_allclose(getattr(particle, name).numpy(),
                                                   getattr(generated[kind], name).numpy(),
                                                   rtol=2e-7, atol=0, err_msg=f"{kind}.{name}")

    def test_reordered_columns_are_dispatched_by_id_for_every_kind(self):
        for kind, source in self.paths.items():
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                lines = source.read_text().splitlines()
                # One row suffices to check scalar and vector reconstruction.
                path = Path(directory) / source.name
                path.write_text(" ".join(reversed(lines[0].split())) + "\n" +
                                " ".join(reversed(lines[1].split())) + "\n")
                particle = input_parser(path)
                expected = input_parser(source)
                for name, _ in INPUT_FIELDS[kind]:
                    np.testing.assert_array_equal(getattr(particle, name).numpy()[0],
                                                  getattr(expected, name).numpy()[0])

    def test_invalid_input_is_rejected_before_allocation(self):
        header = "1 2 3 4 5 6 7 8\n"
        good = "0 0 0 0 0 0 1000 0.001\n"
        cases = ["", "1 1 2 3 4 5 6 7\n" + good,
                 "1 2 3 4 5 6 7 99\n" + good, "1.0 2 3 4 5 6 7 8\n" + good,
                 header + "0 0\n", header + "\n", header + good.replace("1000", "nan"),
                 header + good.replace("1000", "inf"), header + good.replace("1000", "1e100"),
                 header + good.replace("1000", "-1"), header + good.replace("0.001", "1e-100"),
                 header + good.replace("1000", "oops")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input_SPH.txt"
            for text in cases:
                with self.subTest(text=text):
                    path.write_text(text)
                    with self.assertRaises(ValueError):
                        input_parser(path)

    def test_empty_boundary_and_zero_contact_buffers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input_DEMBND.txt"
            path.write_text(" ".join(str(i) for i in range(1, 17)) + "\n")
            boundary = input_parser(path)
            self.assertEqual(boundary.pos.shape[0], 0)
            dem = input_parser(self.paths["DEM"])
            scales = validate_dem_particles(self.runtime(), dem, boundary)
            self.assertEqual(scales[1], 0)
            for kind in ("dem", "bnd"):
                for side in ("old", "new"):
                    self.assertEqual(getattr(dem, f"contact_{kind}_id_{side}").size, 0)
                    np.testing.assert_array_equal(
                        getattr(dem, f"contact_{kind}_offset_{side}").numpy(), 0)

    def test_ghost_mirror_range_and_initial_dirichlet_velocity(self):
        solv = self.runtime(eisph=True)
        fluid, boundary = load_eisph_particles(solv)
        np.testing.assert_array_equal(boundary.vel.numpy(),
                                      2*boundary.vel_bc.numpy()-fluid.vel.numpy()[boundary.mirror.numpy()])
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            directory.joinpath("input_EISPH.txt").write_bytes(self.paths["EISPH"].read_bytes())
            source = self.paths["EISPHBND"].read_text().splitlines()
            row = source[1].split()
            for bad in ("-1", "1.5", str(fluid.pos.shape[0]), "2147483648"):
                row[-1] = bad
                directory.joinpath("input_EISPHBND.txt").write_text(source[0]+"\n"+" ".join(row)+"\n")
                solv.input_dir = str(directory)
                with self.subTest(mirror=bad), self.assertRaises(ValueError):
                    load_eisph_particles(solv)

    def test_loaded_materials_control_contact_dt_and_search_extents(self):
        _, _, dem, boundary = load_sph_particles(self.runtime())
        dem.h = wp.array([0.04, 0.12], dtype=float, device="cpu")
        dem.K = wp.array([2e4, 8e4], dtype=float, device="cpu")
        dem.eta = wp.array([4.0, 7.0], dtype=float, device="cpu")
        dem.mu = wp.array([0.1, 0.6], dtype=float, device="cpu")
        with tempfile.TemporaryDirectory() as directory:
            path = write_particle_file(Path(directory)/"input_DEM.txt", "DEM", dem)
            loaded = input_parser(path)
            for name in ("K", "eta", "mu", "h"):
                np.testing.assert_array_equal(getattr(loaded, name).numpy(), getattr(dem, name).numpy())
            scales = dem_search_scales(loaded, boundary)
            self.assertAlmostEqual(scales[2], 0.12)
            self.assertEqual(scales[3], 0.0)
            solv = self.runtime()
            solv.dt = 0.5
            with self.assertRaisesRegex(ValueError, "loaded DEM contact limit"):
                validate_dem_particles(solv, loaded, boundary)

    def test_file_and_generated_particles_have_same_coupled_trajectory(self):
        for device in self.devices:
            with self.subTest(device=device), wp.ScopedDevice(device):
                solv = self.runtime(device)
                self.generation.device = device
                direct = (*DamPtlGeneration(self.generation).build(),
                          *DEMPtlGeneration(self.generation).build())
                loaded = load_sph_particles(solv)
                for particles in (direct, loaded):
                    grids = [wp.HashGrid(32, 32, 32) for _ in range(4)]
                    scales = dem_search_scales(particles[2], particles[3])
                    for grid, p, radius in zip(grids, particles,
                                              (solv.support, solv.support, 2*scales[2], 2*scales[1])):
                        grid.build(p.pos, radius)
                    for step in range(3):
                        grids[0].build(particles[0].pos, solv.support)
                        grids[2].build(particles[2].pos, 2*scales[2])
                        SPHDEM_OneStep(solv, *particles, *grids, step, scales)
                for p, q in zip(direct, loaded):
                    for name in ("pos", "vel", "rho"):
                        np.testing.assert_allclose(getattr(p, name).numpy(), getattr(q, name).numpy(),
                                                   rtol=5e-6, atol=1e-6)
                self.generation.device = "cpu"

    def test_file_and_generated_eisph_have_same_trajectory(self):
        for device in self.devices:
            with self.subTest(device=device):
                solv = self.runtime(device, True)
                self.cavity.device = device
                direct = CavityPtlGeneration(self.cavity).build()
                loaded = load_eisph_particles(solv)
                run_eisph(solv, *direct)
                run_eisph(solv, *loaded)
                for p, q in zip(direct, loaded):
                    for name in ("pos", "vel", "pres"):
                        np.testing.assert_allclose(getattr(p, name).numpy(), getattr(q, name).numpy(),
                                                   rtol=5e-6, atol=1e-6)
                self.cavity.device = "cpu"

    def test_variable_dem_h_preserves_weighted_drag_reaction(self):
        for device in self.devices:
            with self.subTest(device=device), wp.ScopedDevice(device):
                solv = self.runtime(device)
                fluid, wall, dem, _ = load_sph_particles(solv)
                dem.h = wp.array([0.04, 0.10], dtype=float)
                fluid.vel.fill_(wp.vec3(0.5, -0.2, 0.3))
                grids = [wp.HashGrid(32, 32, 32) for _ in range(3)]
                for grid, p in zip(grids, (fluid, wall, dem)):
                    grid.build(p.pos, 0.2)
                wp.launch(Kernel_prep_sphdem, dim=fluid.pos.shape[0],
                          inputs=[fluid, wall, dem, *(g.id for g in grids), 0.2, 0.0,
                                  solv.support, solv.h, solv.dem_porosity_min, solv.dem_porosity_max])
                # Independent NumPy sums check heterogeneous porosity, including
                # a fluid normalization centred at i for each neighbour's h.
                positions, centers = fluid.pos.numpy(), dem.pos.numpy()
                volumes = fluid.m.numpy()/fluid.rho.numpy()
                sample = np.arange(0, len(positions), 17)
                solid_fraction = np.zeros(len(sample))

                def weight(distance, h):
                    q = distance/h
                    return np.where(q < 2, 21/(16*np.pi*h**3)*(1-q/2)**4*(1+2*q), 0)

                for b, h in enumerate(dem.h.numpy()):
                    norm = (weight(np.linalg.norm(positions[sample, None]-positions[None], axis=2), h)
                            * volumes[None]).sum(axis=1)
                    solid_fraction += dem.volume.numpy()[b]*weight(
                        np.linalg.norm(positions[sample]-centers[b], axis=1), h)/(norm+1e-20)
                np.testing.assert_allclose(fluid.porosity.numpy()[sample],
                                           np.clip(1-solid_fraction, solv.dem_porosity_min,
                                                   solv.dem_porosity_max), rtol=2e-6, atol=2e-6)
                wp.launch(Kernel_interaction_dem, dim=dem.pos.shape[0],
                          inputs=[fluid, dem, grids[0].id, grids[2].id, solv.mu,
                                  solv.dem_porosity_min, solv.dem_porosity_max, solv.dt])
                wp.launch(Kernel_interaction_sph, dim=fluid.pos.shape[0],
                          inputs=[fluid, dem, grids[0].id, grids[2].id, 0.2])
                self.assertTrue(np.all(dem.flt_s.numpy() > 0))
                reaction = np.sum((fluid.m.numpy()*fluid.porosity.numpy())[:, None]
                                  * fluid.acc_dem.numpy(), axis=0)
                np.testing.assert_allclose(reaction, -dem.drag.numpy().sum(axis=0),
                                           rtol=2e-5, atol=1e-8)
                # Changing h must change the interpolation weights, not merely storage.
                previous = dem.flt_s.numpy().copy()
                dem.h.fill_(0.07)
                wp.launch(Kernel_interaction_dem, dim=dem.pos.shape[0],
                          inputs=[fluid, dem, grids[0].id, grids[2].id, solv.mu,
                                  solv.dem_porosity_min, solv.dem_porosity_max, solv.dt])
                self.assertGreater(float(np.max(np.abs(previous-dem.flt_s.numpy()))), 1e-4)


if __name__ == "__main__":
    unittest.main()

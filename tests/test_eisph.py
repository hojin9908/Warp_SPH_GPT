import tempfile
import unittest
from dataclasses import fields
from pathlib import Path

import numpy as np
import warp as wp
from PIL import Image

from input_gen.config import GenerationConfig as Solv
from input_gen.config import CoupledGenerationConfig as SPHDEMSolv
from input_gen.gen_eisph import CavityPtlGeneration
from kernel.KERNEL_EISPH_BC import Kernel_Dirichlet_BC
from kernel.KERNEL_EISPH_KNL import (
    Kernel_dw_Wendland_2d,
    Kernel_prepare_kgc,
    Kernel_w_Wendland_2d,
    _laplace_weight,
)
from kernel.KERNEL_EISPH_PPE import Kernel_build_ppe_eisph
from kernel.KERNEL_EISPH_force import Kernel_advec_vis_eisph
from output.gif_eisph import save_cavity_gif
from source.EISPH import run_eisph


@wp.kernel
def sample_eisph_kernel(radii: wp.array(dtype=float),
                        h: float,
                        values: wp.array(dtype=float),
                        slopes: wp.array(dtype=float)) -> None:
    """Sample the two-dimensional EISPH kernel and its radial derivative."""
    i = wp.tid()
    values[i] = Kernel_w_Wendland_2d(radii[i], h)
    slopes[i] = Kernel_dw_Wendland_2d(radii[i], h)


@wp.kernel
def sample_laplace_weight(result: wp.array(dtype=float)) -> None:
    """Sample the signed PDF pair coefficient for an inward kernel gradient."""
    result[0] = _laplace_weight(0.04, wp.vec3(0.1, 0.0, 0.0),
                                wp.vec3(-2.0, 0.0, 0.0), 0.01)


class EISPHTests(unittest.TestCase):
    """Check the fixed cavity, one-pass EISPH projection and GIF output."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize Warp once for all CPU and optional CUDA checks."""
        wp.init()

    def test_preprocessor_preserves_original_example_defaults(self) -> None:
        """Preserve the old case defaults in the preprocessor, outside solver Solv."""
        eisph = Solv(device="cpu")
        sphdem = SPHDEMSolv(device="cpu")

        eisph_fields = [(field.name, field.type) for field in fields(Solv)]
        sphdem_fields = [(field.name, field.type) for field in fields(SPHDEMSolv)]
        self.assertEqual(eisph_fields, sphdem_fields)
        self.assertEqual(eisph.dx, 0.04)
        self.assertEqual(eisph.bnd_layer, 3)
        self.assertAlmostEqual(eisph.h, 0.052)
        self.assertAlmostEqual(eisph.support, 0.104)
        self.assertEqual(eisph.rho0, 1.0)
        self.assertEqual(eisph.dt, 2.0e-3)
        self.assertEqual(eisph.n_steps, 5000)
        self.assertEqual(eisph.output_step, 100)
        self.assertFalse(eisph.dem_enable)
        self.assertEqual(eisph.cells, 25)
        self.assertAlmostEqual(eisph.nu, 0.01)
        self.assertEqual(sphdem.dx, 0.02)
        self.assertAlmostEqual(sphdem.h, 0.026)
        self.assertAlmostEqual(sphdem.support, 0.052)
        self.assertEqual(sphdem.rho0, 1000.0)
        self.assertEqual(sphdem.dt, 1.0e-4)
        self.assertEqual(sphdem.n_steps, 9000)
        self.assertEqual(sphdem.output_step, 90)
        self.assertTrue(sphdem.dem_enable)
        eisph.validate_eisph()
        sphdem.validate_sph()
        sphdem.validate_dem()

        scaled = Solv(device="cpu", dx=0.2, h_factor=1.4)
        explicit = SPHDEMSolv(device="cpu", h=0.03, support=0.06)
        self.assertAlmostEqual(scaled.h, 0.28)
        self.assertAlmostEqual(scaled.support, 0.56)
        self.assertAlmostEqual(explicit.h, 0.03)
        self.assertAlmostEqual(explicit.support, 0.06)
        scaled.validate_eisph()

    def test_wendland_is_normalized_in_two_dimensions(self) -> None:
        """Integrate W over a disk and compare dW/dr with finite differences."""
        h = 0.13
        radii = np.linspace(0.0, 2.0 * h, 4001, dtype=np.float32)
        values = wp.zeros(len(radii), dtype=float, device="cpu")
        slopes = wp.zeros(len(radii), dtype=float, device="cpu")
        with wp.ScopedDevice("cpu"):
            wp.launch(sample_eisph_kernel, dim=len(radii),
                      inputs=[wp.array(radii, device="cpu"), h, values, slopes])
        radial_integrand = 2.0 * np.pi * radii.astype(float) * values.numpy()
        integral = np.sum(0.5 * (radial_integrand[:-1] + radial_integrand[1:])
                          * np.diff(radii.astype(float)))
        self.assertAlmostEqual(float(integral), 1.0, places=6)
        numerical = np.gradient(values.numpy().astype(float), radii.astype(float))
        np.testing.assert_allclose(slopes.numpy()[50:-50:100],
                                   numerical[50:-50:100], rtol=0.003)

    def test_generator_builds_mirrored_cavity_ghosts(self) -> None:
        """Check cell centres, one shared struct, velocity BC and mirror IDs."""
        solv = Solv(device="cpu", dx=0.2, bnd_layer=3,
                           n_steps=1, output_step=1)
        generator = CavityPtlGeneration(solv)
        fluid_pos = generator.fluid_particle()
        boundary_pos, vel_bc, mirror = generator.boundary_particle()
        P_sph, P_bnd = generator.build()

        self.assertIs(type(P_sph), type(P_bnd))
        self.assertEqual(fluid_pos.shape, (solv.cells**2, 3))
        np.testing.assert_allclose(fluid_pos[:, 0].min(), 0.5 * solv.dx)
        np.testing.assert_allclose(fluid_pos[:, 0].max(), solv.length - 0.5 * solv.dx)
        self.assertTrue(np.all((mirror >= 0) & (mirror < len(fluid_pos))))
        np.testing.assert_allclose(P_bnd.pos.numpy(), boundary_pos)
        np.testing.assert_allclose(P_bnd.vel_bc.numpy(), vel_bc)
        np.testing.assert_allclose(P_bnd.vel.numpy(), 2.0 * vel_bc)

        top = (boundary_pos[:, 2] > solv.length) \
            & (boundary_pos[:, 0] > 0.0) & (boundary_pos[:, 0] < solv.length)
        self.assertTrue(np.all(vel_bc[top, 0] == solv.lid_velocity))
        self.assertTrue(np.all(vel_bc[~top] == 0.0))

    def test_laplace_weight_uses_pdf_sign(self) -> None:
        """Store A_ij=2*V_j*(r_ij dot grad W_ij)/r^2."""
        result = wp.zeros(1, dtype=float, device="cpu")
        with wp.ScopedDevice("cpu"):
            wp.launch(sample_laplace_weight, dim=1, inputs=[result])
        expected = 2.0 * 0.04 * (0.1 * -2.0) / 0.01
        self.assertAlmostEqual(float(result.numpy()[0]), expected, places=6)
        self.assertLess(float(result.numpy()[0]), 0.0)

    def test_ppe_assembly_uses_pdf_rhs_sign(self) -> None:
        """Store b_i=(rho_0/dt)*div(u_i*) before cavity mean removal."""
        solv = Solv(device="cpu", dx=0.2, bnd_layer=3,
                           n_steps=1, output_step=1, grid_slice=16)
        P_sph, P_bnd = CavityPtlGeneration(solv).build()
        constant_pressure = 3.25
        P_sph.pres.fill_(constant_pressure)
        n_sph = P_sph.pos.shape[0]
        n_bnd = P_bnd.pos.shape[0]
        with wp.ScopedDevice("cpu"):
            grid_sph = wp.HashGrid(solv.grid_slice, solv.grid_slice,
                                   solv.grid_slice)
            grid_bnd = wp.HashGrid(solv.grid_slice, solv.grid_slice,
                                   solv.grid_slice)
            grid_sph.build(points=P_sph.pos, radius=solv.support)
            grid_bnd.build(points=P_bnd.pos, radius=solv.support)
            wp.launch(Kernel_prepare_kgc, dim=n_sph,
                      inputs=[P_sph, P_bnd, grid_sph.id, grid_bnd.id,
                              solv.support, solv.h])
            wp.launch(Kernel_Dirichlet_BC, dim=n_bnd,
                      inputs=[P_sph, P_bnd, 0])
            wp.launch(Kernel_advec_vis_eisph, dim=n_sph,
                      inputs=[P_sph, P_bnd, grid_sph.id, grid_bnd.id,
                              solv.support, solv.h, solv.nu, solv.dt])
            wp.launch(Kernel_Dirichlet_BC, dim=n_bnd,
                      inputs=[P_sph, P_bnd, 1])
            wp.launch(Kernel_build_ppe_eisph, dim=n_sph,
                      inputs=[P_sph, P_bnd, grid_sph.id, grid_bnd.id,
                              solv.support, solv.h, solv.rho0, solv.dt])

        np.testing.assert_allclose(P_sph.bi.numpy(),
                                   solv.rho0 * P_sph.divergence.numpy() / solv.dt,
                                   rtol=2.0e-6, atol=2.0e-5)
        self.assertTrue(np.all(P_sph.Aij.numpy() < 0.0))
        np.testing.assert_allclose(P_sph.Aijpj.numpy(),
                                   constant_pressure * P_sph.Aij.numpy(),
                                   rtol=2.0e-6, atol=2.0e-5)

    def test_pdf_signed_weight_keeps_viscous_diffusion_direction(self) -> None:
        """Diffuse a centre maximum while convection is zero at that point."""
        solv = Solv(device="cpu", dx=0.2, bnd_layer=3,
                           n_steps=1, output_step=1, grid_slice=16)
        P_sph, P_bnd = CavityPtlGeneration(solv).build()
        positions = P_sph.pos.numpy()
        radius2 = (positions[:, 0] - 0.5) ** 2 \
            + (positions[:, 2] - 0.5) ** 2
        velocity = np.zeros_like(positions)
        velocity[:, 0] = -radius2
        P_sph.vel.assign(velocity)
        n_sph = P_sph.pos.shape[0]
        n_bnd = P_bnd.pos.shape[0]

        with wp.ScopedDevice("cpu"):
            grid_sph = wp.HashGrid(solv.grid_slice, solv.grid_slice,
                                   solv.grid_slice)
            grid_bnd = wp.HashGrid(solv.grid_slice, solv.grid_slice,
                                   solv.grid_slice)
            grid_sph.build(points=P_sph.pos, radius=solv.support)
            grid_bnd.build(points=P_bnd.pos, radius=solv.support)
            wp.launch(Kernel_prepare_kgc, dim=n_sph,
                      inputs=[P_sph, P_bnd, grid_sph.id, grid_bnd.id,
                              solv.support, solv.h])
            wp.launch(Kernel_Dirichlet_BC, dim=n_bnd,
                      inputs=[P_sph, P_bnd, 0])
            wp.launch(Kernel_advec_vis_eisph, dim=n_sph,
                      inputs=[P_sph, P_bnd, grid_sph.id, grid_bnd.id,
                              solv.support, solv.h, solv.nu, solv.dt])

        centre = int(np.argmin(radius2))
        self.assertAlmostEqual(float(velocity[centre, 0]), 0.0)
        self.assertLess(float(P_sph.acc.numpy()[centre, 0]), 0.0)
        self.assertLess(float(P_sph.vel_star.numpy()[centre, 0]), 0.0)
        np.testing.assert_allclose(P_sph.acc.numpy()[centre, 0],
                                   -4.0 * solv.nu,
                                   rtol=1.0e-5, atol=1.0e-7)
        np.testing.assert_allclose(P_sph.vel_star.numpy()[centre, 0],
                                   -4.0 * solv.nu * solv.dt,
                                   rtol=1.0e-5, atol=1.0e-7)

    def test_general_velocity_dirichlet_boundary(self) -> None:
        """Apply nonuniform prescribed velocities to vel and vel_star."""
        solv = Solv(device="cpu", dx=0.2, bnd_layer=3,
                           n_steps=1, output_step=1)
        P_sph, P_bnd = CavityPtlGeneration(solv).build()
        n_sph = P_sph.pos.shape[0]
        n_bnd = P_bnd.pos.shape[0]
        fluid_id = np.arange(n_sph, dtype=np.float32)
        bnd_id = np.arange(n_bnd, dtype=np.float32)
        vel = np.stack((0.01 * fluid_id, -0.02 * fluid_id,
                        0.03 * fluid_id), axis=1).astype(np.float32)
        vel_star = (-0.5 * vel).astype(np.float32)
        vel_bc = np.stack((0.2 + 0.001 * bnd_id,
                           -0.1 + 0.002 * bnd_id,
                           0.05 - 0.001 * bnd_id), axis=1).astype(np.float32)
        P_sph.vel.assign(vel)
        P_sph.vel_star.assign(vel_star)
        P_bnd.vel_bc.assign(vel_bc)

        with wp.ScopedDevice("cpu"):
            wp.launch(Kernel_Dirichlet_BC, dim=n_bnd,
                      inputs=[P_sph, P_bnd, 0])
            wp.launch(Kernel_Dirichlet_BC, dim=n_bnd,
                      inputs=[P_sph, P_bnd, 1])

        mirror = P_bnd.mirror.numpy()
        np.testing.assert_allclose(P_bnd.vel.numpy(),
                                   2.0 * vel_bc - vel[mirror])
        np.testing.assert_allclose(P_bnd.vel_star.numpy(),
                                   2.0 * vel_bc - vel_star[mirror])

    def test_cavity_step_is_fixed_finite_and_mirrored(self) -> None:
        """Check fixed positions, finite fields, projection and circulation direction."""
        solv = Solv(device="cpu", dx=0.1, bnd_layer=3,
                           n_steps=20, output_step=20, grid_slice=16)
        P_sph, P_bnd = CavityPtlGeneration(solv).build()
        initial_positions = P_sph.pos.numpy().copy()
        positions, frames, times = run_eisph(solv, P_sph, P_bnd)

        np.testing.assert_array_equal(P_sph.pos.numpy(), initial_positions)
        np.testing.assert_array_equal(positions, initial_positions)
        self.assertEqual(times, [0.0, solv.n_steps * solv.dt])
        self.assertEqual(len(frames), 2)
        for field in (P_sph.vel, P_sph.vel_star, P_sph.acc, P_sph.pres,
                      P_sph.Aij, P_sph.bi, P_sph.Aijpj,
                      P_sph.divergence, P_sph.residual, P_sph.kgc):
            self.assertTrue(np.all(np.isfinite(field.numpy())))
        np.testing.assert_array_equal(P_sph.vel.numpy()[:, 1], 0.0)
        np.testing.assert_allclose(
            P_bnd.vel.numpy(),
            2.0 * P_bnd.vel_bc.numpy() - P_sph.vel.numpy()[P_bnd.mirror.numpy()],
            rtol=1.0e-6, atol=1.0e-7,
        )
        self.assertAlmostEqual(float(P_sph.bi.numpy().mean()), 0.0, places=4)
        self.assertEqual(float(P_sph.pres.numpy()[0]), 0.0)
        predictor_divergence = P_sph.bi.numpy() * solv.dt / solv.rho0
        self.assertLess(float(np.linalg.norm(P_sph.divergence.numpy())),
                        float(np.linalg.norm(predictor_divergence)))

        u = P_sph.vel.numpy()[:, 0].reshape(solv.cells, solv.cells)
        self.assertGreater(float(u[-1, solv.cells // 2]), 0.0)
        self.assertLess(float(u[solv.cells // 2, solv.cells // 2]), 0.0)

    def test_ppe_uses_one_implicit_pressure_update(self) -> None:
        """Match p^(t+1) with the stored old-pressure diagonal relation."""
        solv = Solv(device="cpu", dx=0.2, bnd_layer=3,
                           n_steps=1, output_step=1, grid_slice=16)
        old_pressure = np.linspace(-0.25, 0.25, solv.cells ** 2,
                                   dtype=np.float32)

        def one_step(initial_pressure: np.ndarray):
            """Return fields needed to check one pressure update."""
            P_sph, P_bnd = CavityPtlGeneration(solv).build()
            P_sph.pres.assign(initial_pressure)
            run_eisph(solv, P_sph, P_bnd)
            return P_sph

        P_sph = one_step(old_pressure)
        pressure = P_sph.pres.numpy()
        bi = P_sph.bi.numpy()
        Aij = P_sph.Aij.numpy()
        Aijpj = P_sph.Aijpj.numpy()
        expected = old_pressure.copy()
        active = np.abs(Aij) > 1.0e-12
        self.assertTrue(np.all(Aij[active] < 0.0))
        expected[active] = (bi[active] + Aijpj[active]) / Aij[active]
        gauge = expected[0] if active[0] else 0.0
        expected = expected - gauge
        np.testing.assert_allclose(pressure, expected, rtol=2.0e-6, atol=2.0e-6)

        for name in ("Aij", "bi", "Aijpj"):
            self.assertTrue(hasattr(P_sph, name))
        for name in ("ppe_diag", "ppe_rhs", "ppe_neighbor"):
            self.assertFalse(hasattr(P_sph, name))

        shifted = one_step(old_pressure + 7.5)
        np.testing.assert_allclose(shifted.pres.numpy(), pressure,
                                   rtol=2.0e-5, atol=2.0e-5)
        np.testing.assert_allclose(shifted.vel.numpy(), P_sph.vel.numpy(),
                                   rtol=2.0e-5, atol=2.0e-6)
        self.assertFalse(hasattr(solv, "ppe_iterations"))
        self.assertFalse(hasattr(solv, "ppe_relaxation"))

    def test_re100_cavity_matches_ghia_centerlines(self) -> None:
        """Compare both steady centreline profiles with the Ghia Re=100 data."""
        solv = Solv(device="cpu", n_steps=5000, output_step=5000)
        P_sph, P_bnd = CavityPtlGeneration(solv).build()

        run_eisph(solv, P_sph, P_bnd)

        velocity = P_sph.vel.numpy().reshape(solv.cells, solv.cells, 3)
        centre = solv.cells // 2
        axis = (np.arange(solv.cells, dtype=np.float64) + 0.5) * solv.dx

        # Ghia et al. (1982), Tables I-II, Re=100, excluding wall points.
        ghia_y = np.array([
            0.9766, 0.9688, 0.9609, 0.9531, 0.8516,
            0.7344, 0.6172, 0.5000, 0.4531, 0.2813,
            0.1719, 0.1016, 0.0703, 0.0625, 0.0547,
        ])
        ghia_u = np.array([
            0.84123, 0.78871, 0.73722, 0.68717, 0.23151,
            0.00332, -0.13641, -0.20581, -0.21090, -0.15662,
            -0.10150, -0.06434, -0.04775, -0.04192, -0.03717,
        ])
        ghia_x = np.array([
            0.9688, 0.9609, 0.9531, 0.9453, 0.9063,
            0.8594, 0.8047, 0.5000, 0.2344, 0.2266,
            0.1563, 0.0938, 0.0781, 0.0703, 0.0625,
        ])
        ghia_w = np.array([
            -0.05906, -0.07391, -0.08864, -0.10313, -0.16914,
            -0.22445, -0.24533, 0.05454, 0.17527, 0.17507,
            0.16077, 0.12317, 0.10890, 0.10091, 0.09233,
        ])
        calculated_u = np.interp(ghia_y, axis, velocity[:, centre, 0])
        calculated_w = np.interp(ghia_x, axis, velocity[centre, :, 2])
        error = np.concatenate((calculated_u - ghia_u,
                                calculated_w - ghia_w))
        self.assertLess(float(np.sqrt(np.mean(error ** 2))), 0.015)
        self.assertLess(float(np.max(np.abs(error))), 0.030)

    def test_cpu_and_cuda_take_the_same_step(self) -> None:
        """Compare a short deterministic EISPH run on CPU and CUDA."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA is unavailable")

        results = []
        for device in ("cpu", "cuda:0"):
            solv = Solv(device=device, dx=0.2, bnd_layer=3,
                               n_steps=2, output_step=2, grid_slice=16)
            P_sph, P_bnd = CavityPtlGeneration(solv).build()
            run_eisph(solv, P_sph, P_bnd)
            results.append((P_sph.vel.numpy(), P_sph.pres.numpy()))

        np.testing.assert_allclose(results[0][0], results[1][0],
                                   rtol=2.0e-5, atol=2.0e-7)
        np.testing.assert_allclose(results[0][1], results[1][1],
                                   rtol=2.0e-5, atol=5.0e-6)

    def test_animation_stays_in_a_tests_temporary_directory(self) -> None:
        """Encode two frames under tests and remove them with the temporary directory."""
        solv = Solv(device="cpu", dx=0.2, n_steps=1,
                           output_step=1, gif_fps=4)
        positions = CavityPtlGeneration(solv).fluid_particle()
        zero = np.zeros_like(positions)
        vortex = zero.copy()
        vortex[:, 0] = -(positions[:, 2] - 0.5)
        vortex[:, 2] = positions[:, 0] - 0.5

        tests_dir = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(dir=tests_dir) as temporary:
            destination = Path(temporary) / "cavity.gif"
            saved = save_cavity_gif(positions, [zero, vortex], [0.0, 0.1],
                                    solv, destination)
            self.assertEqual(saved, destination.resolve())
            self.assertTrue(saved.is_file())
            with Image.open(saved) as image:
                self.assertEqual(image.n_frames, 2)


if __name__ == "__main__":
    unittest.main()

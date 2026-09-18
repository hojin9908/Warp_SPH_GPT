import unittest

import numpy as np
import warp as wp

from input.Config_SPH_DEM import Solv
from input.gen_ptl import DamPtlGeneration
from input.gen_dem import DEMPtlGeneration
from kernel.KERNEL_KNL import Kernel_w_Wendland, Kernel_dw_Wendland
from kernel.KERNEL_force import Kernel_force_sph
from kernel.KERNEL_DEM_step import Kernel_step_dem
from kernel.KERNEL_SPHDEM_interaction import Kernel_prep_sphdem, Kernel_interaction_dem, Kernel_interaction_sph
from output.gif_gen import collect_frame, sphere_faces
from output.output import gather_state
from test_dem import scene, run_force_dem, run_bc_dem


@wp.kernel
def sample_wendland(r: wp.array(dtype=float), h: float,
                    w: wp.array(dtype=float), dw: wp.array(dtype=float)):
    """Sample W [1/m^3] and dW/dr [1/m^4] at supplied radii r [m]."""
    i = wp.tid()
    w[i] = Kernel_w_Wendland(r[i], h)
    dw[i] = Kernel_dw_Wendland(r[i], h)


class ThreeDimensionalTests(unittest.TestCase):
    """Check volume normalization, 3D geometry and out-of-plane mechanics / exchange."""

    @classmethod
    def setUpClass(cls):
        """Run physical checks on CPU and on CUDA when available."""
        wp.init()
        cls.devices = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])

    def test_wendland_volume_normalization_and_derivative(self):
        """Integrate W over a 3D ball and compare its radial derivative to finite differences."""
        for device in self.devices:
            with self.subTest(device=device), wp.ScopedDevice(device):
                h = 0.08
                radii = np.linspace(0.0, 2.0*h, 4001, dtype=np.float32)
                w = wp.zeros(len(radii), dtype=float)
                dw = wp.zeros(len(radii), dtype=float)
                wp.launch(sample_wendland, dim=len(radii), inputs=[wp.array(radii), h, w, dw])
                values, slopes = w.numpy(), dw.numpy()
                integrand = 4.0*np.pi*radii.astype(float)**2*values
                integral = np.sum(0.5*(integrand[:-1] + integrand[1:])*np.diff(radii))
                self.assertAlmostEqual(float(integral), 1.0, places=6)
                numerical = np.gradient(values.astype(float), radii.astype(float))
                np.testing.assert_allclose(slopes[50:-50:100], numerical[50:-50:100], rtol=0.002)
                self.assertEqual(values[-1], 0.0)
                self.assertEqual(slopes[0], 0.0)
                self.assertEqual(slopes[-1], 0.0)

    def test_volume_mass_and_five_wall_geometry(self):
        """Check physical 3D mass, all five walls, unique corners and an open top."""
        s = Solv(device="cpu")
        fluid = DamPtlGeneration(s).fluid_particle()
        sph_wall = DamPtlGeneration(s).boundary_particle()
        dem = DEMPtlGeneration(s).dem_particle()
        dem_wall = DEMPtlGeneration(s).boundary_particle()
        # Known initial fluid volume is 0.5*0.5*0.4 m^3, and DEM count stays twenty.
        self.assertAlmostEqual(len(fluid)*s.rho0*s.dx**3, 100.0)
        self.assertEqual(len(dem), 20)
        self.assertEqual(np.linalg.matrix_rank(fluid - fluid.mean(axis=0)), 3)
        self.assertEqual(np.linalg.matrix_rank(dem - dem.mean(axis=0)), 3)
        self.assertAlmostEqual(np.ptp(fluid[:, 1]) + s.dx, s.fluid_depth)
        self.assertAlmostEqual(np.ptp(fluid[:, 2]) + s.dx, s.fluid_height)
        self.assertAlmostEqual(s.dem_mass, s.dem_rho*4*np.pi*s.dem_radius**3/3)
        self.assertAlmostEqual(s.dem_inertia, 2*s.dem_mass*s.dem_radius**2/5)
        for wall in (sph_wall, dem_wall):
            self.assertEqual(len(wall), len(np.unique(wall, axis=0)))
            for axis in (0, 1, 2):
                self.assertLess(wall[:, axis].min(), 0.0)
            self.assertGreaterEqual(wall[:, 0].max(), s.tank_width)
            self.assertGreaterEqual(wall[:, 1].max(), s.tank_depth)
            self.assertGreaterEqual(wall[:, 2].max(), s.tank_height)
            interior = ((wall[:, 0] > 0) & (wall[:, 0] < s.tank_width)
                        & (wall[:, 1] > 0) & (wall[:, 1] < s.tank_depth))
            self.assertTrue(np.all(wall[interior, 2] < 0.0))
        # Reject impossible depth placement and a time step unsafe for the sphere mass.
        with self.assertRaises(ValueError):
            DEMPtlGeneration(Solv(dem_origin_y=0.39)).dem_particle()
        with self.assertRaises(ValueError):
            Solv(dt=1.0e-3).validate_dem()

    def test_gravity_points_along_negative_z(self):
        """SPH acceleration and DEM body force must use only the negative z axis."""
        for device in self.devices:
            with self.subTest(device=device), wp.ScopedDevice(device):
                s, P, B, D, DB, grids = scene(device)
                wp.launch(Kernel_force_sph, dim=P.pos.shape[0],
                          inputs=[P, B, grids[0].id, grids[1].id,
                                  s.support, s.h, s.mu, s.g])
                expected_acc = np.zeros_like(P.acc.numpy())
                expected_acc[:, 2] = -s.g
                np.testing.assert_allclose(P.acc.numpy(), expected_acc, rtol=1e-7)

                run_force_dem(D, grids[2].id, s.dem_radius, s.g, s.dt)
                expected_force = np.zeros_like(D.force.numpy())
                expected_force[:, 2] = -s.dem_mass * s.g
                np.testing.assert_allclose(D.force.numpy(), expected_force, rtol=1e-7)

    def test_oblique_contact_and_three_axis_translation(self):
        """A diagonal normal must generate all three force components and conserve pair force."""
        for device in self.devices:
            with self.subTest(device=device), wp.ScopedDevice(device):
                s, P, B, D, DB, grids = scene(device, dem_nx=2)
                normal = np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)
                positions = np.array([[0.2, 0.12, 0.6], [0.2, 0.12, 0.6] + 0.049*normal], np.float32)
                D.pos.assign(positions)
                D.vel.assign(np.array([normal, -normal], np.float32))
                grids[2].build(D.pos, s.dem_support)
                run_force_dem(D, grids[2].id, s.dem_radius, 0.0, s.dt)
                expected = -(s.dem_K*0.001 + 2*s.dem_eta)*normal
                np.testing.assert_allclose(D.force.numpy()[0], expected, rtol=5e-5, atol=0.002)
                np.testing.assert_allclose(D.force.numpy().sum(axis=0), 0.0, atol=1e-5)
                velocity = D.vel.numpy() + D.force.numpy()/s.dem_mass*s.dt
                wp.launch(Kernel_step_dem, dim=2, inputs=[D, s.dt])
                np.testing.assert_allclose(D.pos.numpy(), positions + velocity*s.dt, atol=1e-7)
                self.assertTrue(np.all(D.pos.numpy()[0] != positions[0]))

    def test_transverse_slip_and_three_axis_spin(self):
        """A y-normal contact with z-slip produces x torque; initial spin accepts all axes."""
        for device in self.devices:
            with self.subTest(device=device), wp.ScopedDevice(device):
                s, P, B, D, DB, grids = scene(device, dem_nx=2,
                                              dem_omega_x=1.0, dem_omega_y=2.0, dem_omega_z=3.0)
                np.testing.assert_allclose(D.omega.numpy(), [[1, 2, 3], [1, 2, 3]])
                D.omega.zero_()
                D.pos.assign(np.array([[0.2, 0.10, 0.6], [0.2, 0.149, 0.6]], np.float32))
                D.vel.assign(np.array([[0, 0, 0], [0, 0, 0.01]], np.float32))
                grids[2].build(D.pos, s.dem_support)
                run_force_dem(D, grids[2].id, s.dem_radius, 0.0, s.dt)
                tangential = s.dem_K*0.01*s.dt + s.dem_eta*0.01
                np.testing.assert_allclose(D.torque.numpy()[:, 0], 0.0245*tangential, rtol=1e-5)
                wp.launch(Kernel_step_dem, dim=2, inputs=[D, s.dt])
                np.testing.assert_allclose(D.omega.numpy()[:, 0], 0.0245*tangential/s.dem_inertia*s.dt, rtol=1e-5)

    def test_front_and_back_wall_contacts(self):
        """Both y-depth walls repel approaching spheres without storing wall loads."""
        for device in self.devices:
            for back in (False, True):
                with self.subTest(device=device, back=back), wp.ScopedDevice(device):
                    s, P, B, D, DB, grids = scene(device, dem_nx=1)
                    sign = -1.0 if back else 1.0
                    y = s.tank_depth - 0.02 if back else 0.02
                    D.pos.fill_(wp.vec3(0.4, y, 0.4))
                    D.vel.fill_(wp.vec3(0.1, -sign*0.2, 0.0))
                    grids[2].build(D.pos, s.dem_support)
                    wall_pos = DB.pos.numpy().copy()
                    run_bc_dem(D, DB, grids[3].id, s.dem_bnd_radius, s.dt)
                    self.assertGreater(sign*D.force.numpy()[0, 1], 0.0)
                    self.assertGreater(np.linalg.norm(D.torque.numpy()[0]), 0.0)
                    np.testing.assert_array_equal(DB.pos.numpy(), wall_pos)
                    np.testing.assert_array_equal(DB.vel.numpy(), 0.0)
                    for field in ("acc", "force", "torque"):
                        self.assertFalse(hasattr(DB, field))

    def test_pressure_gradient_and_drag_in_xyz(self):
        """Check an arbitrary 3D pressure gradient and drag / weighted reaction in all axes."""
        for device in self.devices:
            with self.subTest(device=device), wp.ScopedDevice(device):
                s, P, B, D, DB, grids = scene(device, dem_nx=1)
                gradient = np.array([400.0, -900.0, 700.0])
                for part in (P, B):
                    part.pres.assign((10000.0 + part.pos.numpy() @ gradient).astype(np.float32))
                wp.launch(Kernel_prep_sphdem, dim=P.pos.shape[0],
                          inputs=[P, B, D, *(grid.id for grid in grids[:3]), s.dem_support,
                                  s.dem_h, s.support, s.h, s.dem_porosity_min, s.dem_porosity_max])
                center = np.argmin(np.linalg.norm(P.pos.numpy() - [0.2, 0.2, 0.16], axis=1))
                np.testing.assert_allclose(P.pgf.numpy()[center], -gradient, rtol=0.06)
                D.pos.fill_(wp.vec3(0.2, 0.2, 0.16))
                grids[2].build(D.pos, s.dem_support)
                P.pgf.fill_(wp.vec3(*(-gradient)))
                fluid_velocity = np.array([0.2, -0.1, 0.3])
                P.vel.fill_(wp.vec3(*fluid_velocity))
                P.porosity.fill_(0.8)
                wp.launch(Kernel_interaction_dem, dim=1,
                          inputs=[P, D, grids[0].id, grids[2].id, s.dem_support, s.dem_h,
                                  s.mu, s.dem_porosity_min, s.dem_porosity_max, s.dt])
                np.testing.assert_allclose(D.pressure_force.numpy()[0], -s.dem_volume*gradient, rtol=3e-6)
                drag = D.drag.numpy()[0]
                np.testing.assert_allclose(drag/np.linalg.norm(drag), fluid_velocity/np.linalg.norm(fluid_velocity), rtol=3e-6)
                wp.launch(Kernel_interaction_sph, dim=P.pos.shape[0],
                          inputs=[P, D, grids[0].id, grids[2].id, s.dem_support, s.dem_h])
                reaction = np.sum(P.m.numpy()[:, None]*P.porosity.numpy()[:, None]*P.acc_dem.numpy(), axis=0)
                np.testing.assert_allclose(reaction, -drag, rtol=3e-6, atol=1e-6)

    def test_output_retains_depth_and_physical_spheres(self):
        """Verify GIF input keeps y-depth, physical spheres, and 3D VTK mass."""
        with wp.ScopedDevice("cpu"):
            s, P, B, D, DB, grids = scene("cpu", dem_ny=2, dem_origin_y=0.1)
            frame = collect_frame(P, B, 0.0, D, DB)
            for name in ("pos_sph", "pos_bnd", "pos_dem", "pos_dem_bnd"):
                self.assertEqual(frame[name].shape[1], 3)
                self.assertGreater(np.ptp(frame[name][:, 1]), 0.0)
            vertices = sphere_faces().reshape(-1, 3)
            np.testing.assert_allclose(np.linalg.norm(vertices, axis=1), 1.0, atol=1e-14)
            np.testing.assert_allclose(np.ptp(vertices, axis=0), [2, 2, 2], atol=1e-14)
            state = gather_state(P, B)
            np.testing.assert_allclose(state['m'], s.rho0*s.dx**3, rtol=1e-7)


if __name__ == "__main__":
    unittest.main()

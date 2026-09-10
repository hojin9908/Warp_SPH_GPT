import contextlib
import io
import os
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import warp as wp
from PIL import Image

from input.Config import Solv
from input.gen_ptl import DamPtlGeneration
from input.gen_dem import DEMPtlGeneration
from kernel.KERNEL_DEM_force import Kernel_count_dem_contacts, Kernel_force_dem
from kernel.KERNEL_DEM_BC import (
    Kernel_reset_dem_bnd, Kernel_count_bnd_contacts, Kernel_bc_dem, Kernel_acc_dem_bnd)
from kernel.KERNEL_DEM_step import Kernel_step_dem
from kernel.KERNEL_SPHDEM_interaction import (
    DEM_drag_beta, Kernel_prep_sphdem, Kernel_interaction_dem, Kernel_interaction_sph)
from source.Simulation import SPH_OneStep, SPHDEM_OneStep
from main import run_forward


@wp.kernel
def sample_beta(eps: wp.array(dtype=float), speed: wp.array(dtype=float),
                result: wp.array(dtype=float)):
    """
    Evaluate the drag helper at independent sample points for CPU / CUDA tests.

    eps: sample porosities [N] [-]
    speed: sample relative speeds [N] [m/s]

    # Output
    result[a]: beta / (1 - eps) for rho=1000, mu=0.05 and diameter=0.05.
    """
    a = wp.tid()
    result[a] = DEM_drag_beta(eps[a], 1000.0, 0.05, 0.05, speed[a])


def scene(device, **kwargs):
    """
    Build a complete dam-break fixture on the requested test device.

    device: Warp device name (cpu or cuda:0)
    kwargs: Solv overrides for an individual physical test

    return: solv, SPH, SPH boundary, DEM, DEM boundary, grids
    Grid order is SPH / SPH boundary / DEM / DEM boundary and matches the
    SPHDEM_OneStep arguments. Each grid is built from the initial positions.
    """
    # A compact 3D fluid volume keeps analytic tests fast on CPU and CUDA.
    settings = dict(dx=0.04, h=0.052, support=0.104, tank_width=0.8,
                    tank_height=0.8, tank_depth=0.32, fluid_width=0.4,
                    fluid_height=0.4, fluid_depth=0.32, dem_nx=2,
                    dem_ny=1, dem_nz=1, dem_origin_z=0.16, grid_slice=32)
    settings.update(kwargs)
    solv = Solv(device=device, **settings)
    P, B = DamPtlGeneration(solv).build()
    D, DB = DEMPtlGeneration(solv).build()
    # Use the same four independent neighbour sets as the production solver.
    grids = [wp.HashGrid(32, 32, 32, device=device) for _ in range(4)]
    for grid, part, radius in zip(grids, (P, B, D, DB),
                                  (solv.support, solv.support, solv.dem_support, 2 * solv.dem_bnd_radius)):
        grid.build(part.pos, radius)
    return solv, P, B, D, DB, grids


def active_contact_count(P_dem, kind):
    """Return logical directed contact count E from the old/read CSR side."""
    offsets = getattr(P_dem, f"contact_{kind}_offset_old").numpy()
    return int(offsets[-1])


def contact_history(P_dem, kind, subject, neighbour):
    """Return one history by stable pair IDs without assuming compact order."""
    offsets = getattr(P_dem, f"contact_{kind}_offset_old").numpy()
    ids = getattr(P_dem, f"contact_{kind}_id_old").numpy()
    tangent = getattr(P_dem, f"tang_{kind}_old").numpy()
    for index in range(int(offsets[subject]), int(offsets[subject + 1])):
        if int(ids[index]) == neighbour:
            return tangent[index]
    raise KeyError((kind, subject, neighbour))


def contact_storage_nbytes(P_dem):
    """Return resident bytes belonging to all old/read and new/write CSR fields."""
    names = ("contact_dem_offset_old", "contact_dem_id_old", "tang_dem_old",
             "contact_dem_offset_new", "contact_dem_id_new", "tang_dem_new",
             "contact_bnd_offset_old", "contact_bnd_id_old", "tang_bnd_old",
             "contact_bnd_offset_new", "contact_bnd_id_new", "tang_bnd_new")
    return sum(getattr(P_dem, name).capacity for name in names)


def run_force_dem(P_dem, grid_dem, radius_max, g, dt):
    """Rebuild and swap the DEM CSR using fields owned by P_dem."""
    n_dem = P_dem.pos.shape[0]
    P_dem.contact_dem_offset_new.zero_()
    wp.launch(Kernel_count_dem_contacts, dim=n_dem,
              inputs=[P_dem, grid_dem, radius_max])
    wp.utils.array_scan(P_dem.contact_dem_offset_new,
                        P_dem.contact_dem_offset_new, inclusive=False)
    contact_count = int(
        P_dem.contact_dem_offset_new[n_dem:n_dem + 1].numpy()[0])
    if contact_count < 0:
        raise OverflowError("directed DEM contact count exceeded int32 CSR capacity")
    P_dem.contact_dem_id_new = wp.empty(
        contact_count, dtype=wp.int32, device=P_dem.pos.device)
    P_dem.tang_dem_new = wp.empty(
        contact_count, dtype=wp.vec3, device=P_dem.pos.device)
    wp.launch(Kernel_force_dem, dim=P_dem.pos.shape[0],
              inputs=[P_dem, grid_dem, radius_max, g, dt])
    P_dem.contact_dem_offset_old, P_dem.contact_dem_offset_new = (
        P_dem.contact_dem_offset_new, P_dem.contact_dem_offset_old)
    P_dem.contact_dem_id_old, P_dem.contact_dem_id_new = (
        P_dem.contact_dem_id_new, P_dem.contact_dem_id_old)
    P_dem.tang_dem_old, P_dem.tang_dem_new = (
        P_dem.tang_dem_new, P_dem.tang_dem_old)
    return contact_count


def run_bc_dem(P_dem, P_dem_bnd, grid_dem_bnd, radius_max, dt):
    """Rebuild and swap the boundary CSR using fields owned by P_dem."""
    n_dem = P_dem.pos.shape[0]
    P_dem.contact_bnd_offset_new.zero_()
    wp.launch(Kernel_count_bnd_contacts, dim=n_dem,
              inputs=[P_dem, P_dem_bnd, grid_dem_bnd, radius_max])
    wp.utils.array_scan(P_dem.contact_bnd_offset_new,
                        P_dem.contact_bnd_offset_new, inclusive=False)
    contact_count = int(
        P_dem.contact_bnd_offset_new[n_dem:n_dem + 1].numpy()[0])
    if contact_count < 0:
        raise OverflowError("DEM-boundary contact count exceeded int32 CSR capacity")
    P_dem.contact_bnd_id_new = wp.empty(
        contact_count, dtype=wp.int32, device=P_dem.pos.device)
    P_dem.tang_bnd_new = wp.empty(
        contact_count, dtype=wp.vec3, device=P_dem.pos.device)
    wp.launch(Kernel_bc_dem, dim=P_dem.pos.shape[0],
              inputs=[P_dem, P_dem_bnd, grid_dem_bnd, radius_max, dt])
    P_dem.contact_bnd_offset_old, P_dem.contact_bnd_offset_new = (
        P_dem.contact_bnd_offset_new, P_dem.contact_bnd_offset_old)
    P_dem.contact_bnd_id_old, P_dem.contact_bnd_id_new = (
        P_dem.contact_bnd_id_new, P_dem.contact_bnd_id_old)
    P_dem.tang_bnd_old, P_dem.tang_bnd_new = (
        P_dem.tang_bnd_new, P_dem.tang_bnd_old)
    return contact_count


class DEMTests(unittest.TestCase):
    """Check DEM contact mechanics, two-phase force exchange and separated output."""

    @classmethod
    def setUpClass(cls):
        """Initialize Warp once and repeat physical checks on every available test device."""
        wp.init()
        cls.devices = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])

    def test_sph_coupling_field_initialization(self):
        """Check SPH owns initialized exchange fields in both SPH-only and coupled modes."""
        for device in self.devices:
            for enabled in (False, True):
                with self.subTest(device=device, dem_enable=enabled), wp.ScopedDevice(device):
                    # No DEM generator is needed to allocate fields belonging to the fluid.
                    s = Solv(device=device, dem_enable=enabled)
                    P, B = DamPtlGeneration(s).build()
                    self.assertEqual(P.porosity.shape, P.rho.shape)
                    self.assertEqual(P.pgf.shape, P.acc.shape)
                    self.assertEqual(P.acc_dem.shape, P.acc.shape)
                    np.testing.assert_array_equal(P.porosity.numpy(), 1.0)
                    np.testing.assert_array_equal(P.pgf.numpy(), 0.0)
                    np.testing.assert_array_equal(P.acc_dem.numpy(), 0.0)
                    # The diagnostic reaction must not alias the total acceleration.
                    P.acc_dem.fill_(wp.vec3(1.0, 2.0, 0.0))
                    np.testing.assert_array_equal(P.acc.numpy(), 0.0)

    def test_sparse_history_allocation_replaces_dense_pairs(self):
        """Empty old/new CSR sides replace dense pair-history matrices."""
        with wp.ScopedDevice("cpu"):
            s = Solv(device="cpu")
            D, DB = DEMPtlGeneration(s).build()
            n_dem, n_bnd = D.pos.shape[0], DB.pos.shape[0]
            dense_bytes = 12 * n_dem * (n_dem + n_bnd)
            for kind in ("dem", "bnd"):
                for side in ("old", "new"):
                    offset = getattr(D, f"contact_{kind}_offset_{side}")
                    ids = getattr(D, f"contact_{kind}_id_{side}")
                    tangent = getattr(D, f"tang_{kind}_{side}")
                    self.assertEqual(offset.shape, (n_dem + 1,))
                    np.testing.assert_array_equal(offset.numpy(), 0)
                    self.assertEqual(ids.shape, (0,))
                    self.assertEqual(tangent.shape, (0,))
            self.assertFalse(hasattr(D, "contact_dem_offset"))
            self.assertFalse(hasattr(D, "contact_bnd_offset"))
            self.assertFalse(hasattr(D, "contact_overflow"))
            self.assertFalse(hasattr(s, "dem_contact_slots"))
            self.assertEqual(contact_storage_nbytes(D), 16 * (n_dem + 1))
            self.assertLess(contact_storage_nbytes(D), dense_bytes // 100)

    def test_compact_csr_rows_store_all_stable_directed_ids(self):
        """Scanned row offsets delimit every active directed stable ID exactly."""
        for device in self.devices:
            with self.subTest(device=device), wp.ScopedDevice(device):
                s, P, B, D, DB, grids = scene(device, dem_nx=3)
                D.pos.fill_(wp.vec3(0.3, 0.6, 0.16))
                grids[2].build(D.pos, s.dem_support)
                count = run_force_dem(D, grids[2].id, s.dem_radius, 0.0, s.dt)
                offsets = D.contact_dem_offset_old.numpy()
                ids = D.contact_dem_id_old.numpy()
                self.assertEqual(count, 6)  # three particles, two directed neighbours each
                self.assertEqual(int(offsets[-1]), count)
                self.assertEqual(D.contact_dem_id_old.shape, (count,))
                self.assertEqual(D.tang_dem_old.shape, (count,))
                np.testing.assert_array_equal(np.diff(offsets), [2, 2, 2])
                for a in range(3):
                    row = ids[int(offsets[a]):int(offsets[a + 1])]
                    self.assertEqual(set(map(int, row)), {0, 1, 2} - {a})

    def test_csr_grows_shrinks_to_zero_and_recontact_starts_fresh(self):
        """Exact-E arrays follow 0→2→6→2→0→2 and preserve only survivors."""
        for device in self.devices:
            with self.subTest(device=device), wp.ScopedDevice(device):
                s, P, B, D, DB, grids = scene(device, dem_nx=3)
                self.assertEqual(
                    run_force_dem(D, grids[2].id, s.dem_radius, 0.0, s.dt), 0)
                self.assertEqual(D.contact_dem_id_old.shape, (0,))
                self.assertEqual(D.tang_dem_old.shape, (0,))
                self.assertEqual(contact_storage_nbytes(D), 16 * (3 + 1))

                pair = np.array([[0.200, 0.60, 0.16],
                                 [0.249, 0.60, 0.16],
                                 [0.600, 0.60, 0.16]], dtype=np.float32)
                D.pos.assign(pair)
                D.vel.assign(np.array([[0, 0, 0], [0, 0.01, 0], [0, 0, 0]],
                                      dtype=np.float32))
                grids[2].build(D.pos, s.dem_support)
                self.assertEqual(
                    run_force_dem(D, grids[2].id, s.dem_radius, 0.0, s.dt), 2)
                first = contact_history(D, "dem", 0, 1).copy()
                self.assertEqual(D.contact_dem_id_old.shape, (2,))
                self.assertEqual(D.contact_dem_id_new.shape, (0,))
                self.assertEqual(contact_storage_nbytes(D), 16 * (3 + 1) + 16 * 2)

                D.pos.fill_(wp.vec3(0.3, 0.6, 0.16))
                grids[2].build(D.pos, s.dem_support)
                self.assertEqual(
                    run_force_dem(D, grids[2].id, s.dem_radius, 0.0, s.dt), 6)
                self.assertEqual(D.contact_dem_id_old.shape, (6,))
                self.assertEqual(D.tang_dem_old.shape, (6,))
                self.assertEqual(D.contact_dem_id_new.shape, (2,))
                self.assertEqual(contact_storage_nbytes(D),
                                 16 * (3 + 1) + 16 * (2 + 6))

                # Shrink to one surviving undirected pair. Its two directed
                # histories must survive exact reallocation rather than reset.
                D.pos.assign(pair)
                grids[2].build(D.pos, s.dem_support)
                self.assertEqual(
                    run_force_dem(D, grids[2].id, s.dem_radius, 0.0, s.dt), 2)
                np.testing.assert_allclose(
                    contact_history(D, "dem", 0, 1), 3.0 * first,
                    rtol=1e-6, atol=1e-9)
                self.assertEqual(D.contact_dem_id_new.shape, (6,))
                self.assertEqual(contact_storage_nbytes(D),
                                 16 * (3 + 1) + 16 * (6 + 2))

                D.pos.assign(np.array([[0.1, 0.6, 0.16],
                                       [0.4, 0.6, 0.16],
                                       [0.7, 0.6, 0.16]], dtype=np.float32))
                grids[2].build(D.pos, s.dem_support)
                self.assertEqual(
                    run_force_dem(D, grids[2].id, s.dem_radius, 0.0, s.dt), 0)
                self.assertEqual(D.contact_dem_id_old.shape, (0,))
                self.assertEqual(D.tang_dem_old.shape, (0,))
                self.assertEqual(D.contact_dem_id_new.shape, (2,))
                self.assertEqual(contact_storage_nbytes(D),
                                 16 * (3 + 1) + 16 * 2)

                D.pos.assign(pair)
                grids[2].build(D.pos, s.dem_support)
                self.assertEqual(
                    run_force_dem(D, grids[2].id, s.dem_radius, 0.0, s.dt), 2)
                np.testing.assert_allclose(
                    contact_history(D, "dem", 0, 1), first,
                    rtol=1e-6, atol=1e-9)
                self.assertEqual(D.contact_dem_id_new.shape, (0,))
                self.assertEqual(contact_storage_nbytes(D), 16 * (3 + 1) + 16 * 2)

    def test_coupled_step_publishes_exact_current_csr(self):
        """Production publishes exact current CSR and retains it across CUDA steps."""
        for device in self.devices:
            with self.subTest(device=device), wp.ScopedDevice(device):
                s, P, B, D, DB, grids = scene(device, dem_nx=3, g=0.0)
                pair = np.array([[0.200, 0.60, 0.16],
                                 [0.249, 0.60, 0.16],
                                 [0.600, 0.60, 0.16]], dtype=np.float32)
                velocity = np.array([[0, 0, 0], [0, 0.01, 0], [0, 0, 0]],
                                    dtype=np.float32)
                D.pos.assign(pair)
                D.vel.assign(velocity)
                grids[2].build(D.pos, s.dem_support)
                SPHDEM_OneStep(s, P, B, D, DB, *grids, step=0)
                self.assertEqual(active_contact_count(D, "dem"), 2)
                self.assertEqual(D.contact_dem_id_old.shape, (2,))
                self.assertEqual(D.tang_dem_old.shape, (2,))
                first = contact_history(D, "dem", 0, 1).copy()

                # Follow the same rebuild contract as run_forward, while
                # holding geometry/velocity fixed to isolate history lifetime.
                D.pos.assign(pair)
                D.vel.assign(velocity)
                D.omega.zero_()
                grids[0].build(P.pos, s.support)
                grids[2].build(D.pos, s.dem_support)
                SPHDEM_OneStep(s, P, B, D, DB, *grids, step=1)
                self.assertEqual(active_contact_count(D, "dem"), 2)
                np.testing.assert_allclose(
                    contact_history(D, "dem", 0, 1), 2.0 * first,
                    rtol=1e-6, atol=1e-9)

    def test_csr_replaces_one_neighbour_without_losing_survivor_history(self):
        """A rebuilt row can release one ID, keep one, and add another safely."""
        for device in self.devices:
            with self.subTest(device=device), wp.ScopedDevice(device):
                s, P, B, D, DB, grids = scene(device, dem_nx=4)
                first_positions = np.array([
                    [0.300, 0.600, 0.16],
                    [0.349, 0.600, 0.16],
                    [0.251, 0.600, 0.16],
                    [0.600, 0.700, 0.16],
                ], dtype=np.float32)
                D.pos.assign(first_positions)
                D.vel.assign(np.array([
                    [0, 0, 0], [0, 0, 0], [0, 0.01, 0], [0.01, 0, 0]
                ], dtype=np.float32))
                grids[2].build(D.pos, s.dem_support)
                self.assertEqual(
                    run_force_dem(D, grids[2].id, s.dem_radius, 0.0, s.dt), 4)
                first = contact_history(D, "dem", 0, 2).copy()
                offsets = D.contact_dem_offset_old.numpy()
                ids = D.contact_dem_id_old.numpy()
                self.assertEqual(set(ids[offsets[0]:offsets[1]]), {1, 2})

                second_positions = first_positions.copy()
                second_positions[1] = [0.600, 0.600, 0.16]
                second_positions[3] = [0.300, 0.649, 0.16]
                D.pos.assign(second_positions)
                grids[2].build(D.pos, s.dem_support)
                self.assertEqual(
                    run_force_dem(D, grids[2].id, s.dem_radius, 0.0, s.dt), 4)
                offsets = D.contact_dem_offset_old.numpy()
                ids = D.contact_dem_id_old.numpy()
                row = ids[offsets[0]:offsets[1]]
                self.assertEqual(set(row), {2, 3})
                self.assertEqual(len(row[row == 2]), 1)
                np.testing.assert_allclose(
                    contact_history(D, "dem", 0, 2), 2.0*first,
                    rtol=1e-6, atol=1e-9)

    def test_contact_history_survives_hashgrid_rebuild(self):
        """A persistent pair keeps its history when rebuilt grid order changes."""
        for device in self.devices:
            with self.subTest(device=device), wp.ScopedDevice(device):
                s, P, B, D, DB, grids = scene(device, dem_nx=3)
                D.pos.assign(np.array([[0.200, 0.60, 0.16],
                                       [0.249, 0.60, 0.16],
                                       [0.600, 0.60, 0.16]], dtype=np.float32))
                D.vel.assign(np.array([[0, 0.00, 0],
                                       [0, 0.01, 0],
                                       [0, 0.00, 0]], dtype=np.float32))
                grids[2].build(D.pos, s.dem_support)
                self.assertEqual(
                    run_force_dem(D, grids[2].id, s.dem_radius, 0.0, s.dt), 2)
                first = contact_history(D, "dem", 0, 1).copy()

                # Translate the surviving pair and move the third particle so
                # HashGrid is rebuilt without changing the stable pair IDs.
                D.pos.assign(np.array([[0.401, 0.60, 0.16],
                                       [0.450, 0.60, 0.16],
                                       [0.100, 0.72, 0.16]], dtype=np.float32))
                grids[2].build(D.pos, s.dem_support)
                self.assertEqual(
                    run_force_dem(D, grids[2].id, s.dem_radius, 0.0, s.dt), 2)
                second = contact_history(D, "dem", 0, 1)
                np.testing.assert_allclose(second, 2.0 * first, rtol=1e-6, atol=1e-9)

    def test_normal_contact_and_release_history(self):
        """Check analytic normal force, pair symmetry and history removal after separation."""
        for device in self.devices:
            with self.subTest(device=device), wp.ScopedDevice(device):
                s, P, B, D, DB, grids = scene(device, dem_nx=2, dem_ny=1)
                D.pos.assign(np.array([[0.2, 0.6, 0], [0.249, 0.6, 0]], dtype=np.float32))
                D.vel.assign(np.array([[1, 0, 0], [-1, 0, 0]], dtype=np.float32))
                grids[2].build(D.pos, s.dem_support)
                run_force_dem(D, grids[2].id, s.dem_radius, 0.0, s.dt)
                # overlap=1 mm, approach=2 m/s: spring plus damping along x.
                expected = s.dem_K * 0.001 + s.dem_eta * 2
                np.testing.assert_allclose(D.force.numpy(), [[-expected, 0, 0], [expected, 0, 0]], rtol=2e-5, atol=0.01)
                np.testing.assert_allclose(D.force.numpy().sum(axis=0), 0, atol=1e-6)
                # Place the former neighbour outside the grid query; its history must clear.
                seeded_history = D.tang_dem_old.numpy()
                seeded_history[D.contact_dem_id_old.numpy() >= 0] = [0.0, 0.1, 0.0]
                D.tang_dem_old.assign(seeded_history)
                D.pos.assign(np.array([[0.2, 0.6, 0], [1.5, 0.6, 0]], dtype=np.float32))
                grids[2].build(D.pos, s.dem_support)
                run_force_dem(D, grids[2].id, s.dem_radius, 0.0, s.dt)
                self.assertEqual(active_contact_count(D, "dem"), 0)
                np.testing.assert_array_equal(D.force.numpy(), 0)
                self.assertEqual(D.contact_dem_id_old.shape, (0,))
                self.assertEqual(D.tang_dem_old.shape, (0,))
                # Recontact must start from zero instead of reviving the released value.
                D.pos.assign(np.array([[0.2, 0.6, 0], [0.249, 0.6, 0]], dtype=np.float32))
                D.vel.assign(np.array([[0, 0, 0], [0, 0.01, 0]], dtype=np.float32))
                grids[2].build(D.pos, s.dem_support)
                run_force_dem(D, grids[2].id, s.dem_radius, 0.0, s.dt)
                self.assertAlmostEqual(float(contact_history(D, "dem", 0, 1)[1]),
                                       0.01 * s.dt, places=8)

    def test_friction_history_and_spin(self):
        """Check accumulated sticking displacement, contact torque, spin and sliding friction."""
        for device in self.devices:
            with self.subTest(device=device), wp.ScopedDevice(device):
                s, P, B, D, DB, grids = scene(device, dem_nx=2, dem_ny=1)
                D.pos.assign(np.array([[0.2, 0.6, 0], [0.249, 0.6, 0]], dtype=np.float32))
                D.vel.assign(np.array([[0, 0, 0], [0, 0.01, 0]], dtype=np.float32))
                grids[2].build(D.pos, s.dem_support)
                # Hold geometry fixed to isolate the history spring's growth over two calls.
                for count in (1, 2):
                    run_force_dem(D, grids[2].id, s.dem_radius, 0.0, s.dt)
                    tangential = s.dem_K * 0.01 * s.dt * count + s.dem_eta * 0.01
                    self.assertAlmostEqual(float(D.force.numpy()[0, 1]), tangential, places=5)
                    self.assertAlmostEqual(float(contact_history(D, "dem", 0, 1)[1]),
                                           count * 0.01 * s.dt, places=8)
                # The contact lever arm is R - overlap/2 = 0.0245 m.
                torque = D.torque.numpy()
                np.testing.assert_allclose(torque[:, 2], 0.0245 * tangential, rtol=1e-5)
                wp.launch(Kernel_step_dem, dim=2, inputs=[D, s.dt])
                np.testing.assert_allclose(D.omega.numpy()[:, 2], torque[:, 2] / s.dem_inertia * s.dt, rtol=1e-5)
                # Large slip must reach the reference Coulomb limit.
                D.vel.assign(np.array([[0, 0, 0], [0, 10, 0]], dtype=np.float32))
                D.omega.zero_()
                D.pos.assign(np.array([[0.2, 0.6, 0], [0.249, 0.6, 0]], dtype=np.float32))
                grids[2].build(D.pos, s.dem_support)
                run_force_dem(D, grids[2].id, s.dem_radius, 0.0, s.dt)
                self.assertAlmostEqual(float(D.force.numpy()[0, 1]), s.dem_mu * s.dem_K * 0.001, delta=0.01)

    def test_fixed_boundary_reaction_and_motion(self):
        """Check equal/opposite wall contact, recorded acceleration and unchanged wall motion."""
        for device in self.devices:
            with self.subTest(device=device), wp.ScopedDevice(device):
                s, P, B, D, DB, grids = scene(device, dem_nx=1, dem_ny=1)
                # Enter the contact envelope of the spherical wall's square lattice.
                contact_pos = np.array([[0.4, 0.020, 0.16]], dtype=np.float32)
                contact_vel = wp.vec3(0.1, -0.2, 0)
                D.pos.assign(contact_pos)
                D.vel.fill_(contact_vel)
                grids[2].build(D.pos, s.dem_support)
                run_force_dem(D, grids[2].id, s.dem_radius, 0.0, s.dt)
                wp.launch(Kernel_reset_dem_bnd, dim=DB.pos.shape[0],
                          inputs=[DB, grids[3].id, s.dem_bnd_radius, 0.0])
                # Subtract fixed-fixed preload to isolate the moving sphere's wall reaction.
                fixed_force = DB.force.numpy().copy()
                fixed_pos = DB.pos.numpy().copy()
                bnd_count = run_bc_dem(
                    D, DB, grids[3].id, s.dem_bnd_radius, s.dt)
                bnd_ids = D.contact_bnd_id_old.numpy()
                self.assertGreater(bnd_count, 0)
                self.assertEqual(D.contact_bnd_id_old.shape, (bnd_count,))
                self.assertEqual(D.contact_bnd_id_old.shape, D.tang_bnd_old.shape)
                self.assertEqual(int(D.contact_bnd_offset_old.numpy()[-1]), bnd_count)
                first_neighbour = int(bnd_ids[0])
                first_history = contact_history(D, "bnd", 0, first_neighbour).copy()
                self.assertGreater(np.linalg.norm(first_history), 0)
                wp.launch(Kernel_acc_dem_bnd, dim=DB.pos.shape[0], inputs=[DB])
                self.assertGreater(D.force.numpy()[0, 1], 0)
                np.testing.assert_allclose((DB.force.numpy() - fixed_force).sum(axis=0),
                                           -D.force.numpy()[0], atol=1e-3)
                np.testing.assert_allclose(DB.acc.numpy(), DB.force.numpy() / s.dem_bnd_mass, rtol=1e-6)
                wp.launch(Kernel_step_dem, dim=1, inputs=[D, s.dt])
                np.testing.assert_array_equal(DB.pos.numpy(), fixed_pos)
                np.testing.assert_array_equal(DB.vel.numpy(), 0)
                np.testing.assert_array_equal(DB.omega.numpy(), 0)
                self.assertGreater(np.linalg.norm(DB.torque.numpy()), 0)
                # Clear wall history after leaving contact and the wall query.
                D.pos.fill_(wp.vec3(0.4, 0.8, 0.16))
                grids[2].build(D.pos, s.dem_support)
                run_bc_dem(D, DB, grids[3].id, s.dem_bnd_radius, s.dt)
                self.assertEqual(active_contact_count(D, "bnd"), 0)
                self.assertEqual(D.contact_bnd_id_old.shape, (0,))
                self.assertEqual(D.tang_bnd_old.shape, (0,))
                # Recontact starts fresh; released history is never revived.
                D.pos.assign(contact_pos)
                D.vel.fill_(contact_vel)
                D.omega.zero_()
                grids[2].build(D.pos, s.dem_support)
                run_bc_dem(D, DB, grids[3].id, s.dem_bnd_radius, s.dt)
                np.testing.assert_allclose(
                    contact_history(D, "bnd", 0, first_neighbour),
                    first_history, rtol=1e-6, atol=1e-9)

    def test_drag_branches_and_stokes_limit(self):
        """Check Ergun, both Wen-Yu Reynolds branches and the zero-speed Stokes limit."""
        for device in self.devices:
            with self.subTest(device=device), wp.ScopedDevice(device):
                eps = np.array([0.5, 0.8, 0.9, 1.0, 1.0], np.float32)
                speed = np.array([0.2, 0.1, 0.2, 2.0, 0.0], np.float32)
                result = wp.zeros(5, dtype=float)
                wp.launch(sample_beta, dim=5, inputs=[wp.array(eps), wp.array(speed), result])
                # Independent scalar reference branches with Re = 1000 * speed.
                expected = []
                for e, v in zip(eps.astype(float), speed.astype(float)):
                    if e <= 0.80000002:
                        expected.append(150*(1-e)/e*0.05/0.05**2 + 1.75*1000*v/0.05)
                    elif v == 0:
                        expected.append(18*0.05/0.05**2)
                    else:
                        re = 1000*v
                        cd = 24/re*(1+0.15*re**0.687) if re <= 1000 else 0.44
                        expected.append(0.75*cd*1000*v/0.05*e**-1.65)
                np.testing.assert_allclose(result.numpy(), expected, rtol=2e-6)

    def test_pressure_buoyancy_drag_exchange_and_dry_reset(self):
        """Check pressure interpolation, bounded drag, weighted reaction and dry-load reset."""
        for device in self.devices:
            with self.subTest(device=device), wp.ScopedDevice(device):
                s, P, B, D, DB, grids = scene(device, dem_nx=1, dem_ny=1)
                D.pos.fill_(wp.vec3(0.24, 0.24, 0.16))
                grids[2].build(D.pos, s.dem_support)
                P.vel.fill_(wp.vec3(0.2, 0, 0))
                # A constant hydrostatic -grad(p) gives the exact sphere-volume buoyancy load.
                P.pgf.fill_(wp.vec3(0, s.rho0 * s.g, 0))
                P.porosity.fill_(0.7)
                wp.launch(Kernel_interaction_dem, dim=1,
                          inputs=[P, D, grids[0].id, grids[2].id, s.dem_support, s.dem_h,
                                  s.mu, s.dem_porosity_min, s.dem_porosity_max, s.dt])
                np.testing.assert_allclose(D.pressure_force.numpy()[0], [0, s.dem_volume*s.rho0*s.g, 0], rtol=2e-6)
                self.assertGreater(D.drag.numpy()[0, 0], 0)
                self.assertLess(D.drag.numpy()[0, 0] / s.dem_mass * s.dt, 0.2)
                # Preserve already-computed SPH forces when adding the DEM contribution.
                P.acc.fill_(wp.vec3(1.0, -s.g, 0.0))
                initial_acc = P.acc.numpy().copy()
                wp.launch(Kernel_interaction_sph, dim=P.pos.shape[0],
                          inputs=[P, D, grids[0].id, grids[2].id, s.dem_support, s.dem_h])
                # Reference exchanges momentum using epsilon*m during a fixed-porosity stage.
                reaction = np.sum(P.m.numpy()[:, None] * P.porosity.numpy()[:, None] * P.acc_dem.numpy(), axis=0)
                np.testing.assert_allclose(reaction, -D.drag.numpy().sum(axis=0), rtol=3e-6, atol=1e-5)
                np.testing.assert_allclose(P.acc.numpy(), initial_acc + P.acc_dem.numpy(), rtol=1e-6)
                # Moving above all fluid support must remove the previously stored loads.
                D.pos.fill_(wp.vec3(0.24, 2.0, 0.16))
                grids[2].build(D.pos, s.dem_support)
                wp.launch(Kernel_interaction_dem, dim=1,
                          inputs=[P, D, grids[0].id, grids[2].id, s.dem_support, s.dem_h,
                                  s.mu, s.dem_porosity_min, s.dem_porosity_max, s.dt])
                np.testing.assert_array_equal(D.drag.numpy(), 0)
                np.testing.assert_array_equal(D.pressure_force.numpy(), 0)
                np.testing.assert_array_equal(D.flt_s.numpy(), 0)
                # A dry neighbour contributes zero and overwrites the old diagnostic value.
                previous_acc = P.acc.numpy().copy()
                wp.launch(Kernel_interaction_sph, dim=P.pos.shape[0],
                          inputs=[P, D, grids[0].id, grids[2].id, s.dem_support, s.dem_h])
                np.testing.assert_array_equal(P.acc_dem.numpy(), 0.0)
                np.testing.assert_array_equal(P.acc.numpy(), previous_acc)

    def test_pressure_gradient_sign(self):
        """Check that a hydrostatic pressure field produces an upward -grad(p) in the interior."""
        for device in self.devices:
            with self.subTest(device=device), wp.ScopedDevice(device):
                s, P, B, D, DB, grids = scene(device)
                for part in (P, B):
                    part.pres.assign((10000 - s.rho0*s.g*part.pos.numpy()[:, 1]).astype(np.float32))
                wp.launch(Kernel_prep_sphdem, dim=P.pos.shape[0],
                          inputs=[P, B, D, *(grid.id for grid in grids[:3]),
                                  s.dem_support, s.dem_h, s.support, s.h,
                                  s.dem_porosity_min, s.dem_porosity_max])
                # Interior support avoids a free-surface truncation in this gradient check.
                center = np.argmin(np.linalg.norm(P.pos.numpy()-[0.24, 0.24, 0.16], axis=1))
                np.testing.assert_allclose(P.pgf.numpy()[center], [0, s.rho0*s.g, 0], rtol=0.06, atol=0.02)

    def test_dry_free_fall_and_sph_regression(self):
        """Check gravity is added once and dry DEM does not alter the original SPH trajectory."""
        for device in self.devices:
            with self.subTest(device=device), wp.ScopedDevice(device):
                s, P, B, D, DB, grids = scene(device, dem_origin_y=1.5)
                # Independent SPH-only state supplies the comparison trajectory.
                P0, B0 = DamPtlGeneration(s).build()
                gs = wp.HashGrid(32, 32, 32)
                gb = wp.HashGrid(32, 32, 32)
                gb.build(B0.pos, s.support)
                initial = D.pos.numpy().copy()
                for step in range(5):
                    grids[0].build(P.pos, s.support)
                    grids[2].build(D.pos, s.dem_support)
                    gs.build(P0.pos, s.support)
                    SPH_OneStep(s, P0, B0, gs, gb, step)
                    SPHDEM_OneStep(s, P, B, D, DB, *grids, step)
                np.testing.assert_array_equal(P.pos.numpy(), P0.pos.numpy())
                np.testing.assert_array_equal(P.vel.numpy(), P0.vel.numpy())
                np.testing.assert_allclose(D.vel.numpy()[:, 1], -5*s.g*s.dt, rtol=1e-6)
                # Semi-implicit Euler displacement sums the five updated velocities: 1+...+5.
                np.testing.assert_allclose(D.pos.numpy()[:, 1], initial[:, 1]-s.g*s.dt**2*15, atol=5e-7)

    def test_separate_output_final_frame_and_animation(self):
        """Check phase-specific VTK types, the off-cadence final state and matching GIF frames."""
        with wp.ScopedDevice("cpu"), tempfile.TemporaryDirectory() as tmp:
            s, P, B, D, DB, grids = scene("cpu", n_steps=3, output_step=2,
                                             output_dir=tmp, animation_dir=tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                run_forward(s, P, B, *grids[:2], D, DB, *grids[2:])
            # Three output instants are expected: initial step 0, regular step 2, final step 3.
            for name, types in (("sph", {0, 1}), ("dem", {2, 3})):
                root = ET.parse(os.path.join(tmp, name + ".pvd"))
                datasets = root.findall(".//DataSet")
                self.assertEqual(len(datasets), 3)
                self.assertAlmostEqual(float(datasets[-1].get("timestep")), 3*s.dt)
                data = ET.parse(os.path.join(tmp, datasets[-1].get("file")))
                values = data.find(".//PointData/DataArray[@Name='type']")
                self.assertEqual(set(np.fromstring(values.text, sep=" ").astype(int)), types)
            with Image.open(os.path.join(tmp, "dam_break_sph_dem_3d.gif")) as gif:
                self.assertEqual(gif.n_frames, 3)


if __name__ == "__main__":
    unittest.main()

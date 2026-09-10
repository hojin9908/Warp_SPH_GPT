"""Resolve the existing SPH kernel import spelling without renaming its source file."""

import sys

from kernel import Kernel_KNL

# Existing SPH kernels import KERNEL_KNL, while the tracked file is Kernel_KNL.py.
# Keep those kernels and the original filename untouched on all platforms.
sys.modules[__name__ + ".KERNEL_KNL"] = Kernel_KNL

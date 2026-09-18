# Particle input files

`python -m input_gen.generate` recreates these six files from the original
example presets in `input_gen/config.py`. Run this only when you intend to
replace the inputs. Neither solver entry point generates files automatically.

The first row contains integer column IDs (no text labels). Subsequent rows
contain one particle each, separated by spaces or tabs. IDs are local to each
file, assigned in the Notion structure-table order; vectors expand in x/y/z
order. These are **not the numeric IDs of the original SOPHIA input.txt**.
Every listed column is required. Header/data columns may be reordered together.

| File | Input columns (ID: field) | Default particles |
| --- | --- | ---: |
| input_SPH.txt | `1–3: pos`, `4–6: vel`, `7: rho`, `8: m` | 12,500 |
| input_BND.txt | `1–3: pos`, `4–6: vel`, `7: rho`, `8: m` | 48,336 |
| input_DEM.txt | `1–3: pos`, `4–6: vel`, `7–9: omega`, `10: radius`, `11: rho`, `12: m`, `13: inertia`, `14: K`, `15: eta`, `16: mu`, `17: h` | 20 |
| input_DEMBND.txt | `1–3: pos`, `4–6: vel`, `7–9: omega`, `10: radius`, `11: rho`, `12: m`, `13: inertia`, `14: K`, `15: eta`, `16: mu` | 14,688 |
| input_EISPH.txt | `1–3: pos`, `4–6: vel`, `7: rho`, `8: m` | 625 |
| input_EISPHBND.txt | `1–3: pos`, `4–6: vel_bc`, `7: rho`, `8: m`, `9: mirror` | 336 |

SPH/BND `rho_raw` copies input `rho`; pressure, acceleration and other dynamic
buffers start at zero. Fluid and moving DEM porosity starts at one. DEM volume
is computed as `(4/3)*pi*radius^3`; mass consistency is checked, and inertia is
read as an independent positive material value. The example generator uses
`inertia = 0.4*m*radius^2`. Contact CSR offsets start at zero and edge buffers
are empty. Computed fields have no column IDs.

EISPH fluid `vel_star` copies `vel`. EISPHBND `mirror` is a zero-based row in
input_EISPH.txt. After both files are loaded, ghost velocity is initialized as
`2*vel_bc - fluid.vel[mirror]`. Ghost pressure is supplied by the solver.
The same EISPHptl class represents both EISPH files.

Lengths are m, velocities m/s, angular velocities rad/s, density kg/m³.
SPH/DEM mass is kg; EISPH mass is kg/m (2D mass per unit depth). K is N/m,
eta is N·s/m, mu is dimensionless, and inertia is kg·m².
DEM K/eta/mu/h are stored per moving particle. Fixed DEM boundaries have
K/eta/mu metadata, but the existing contact law uses the moving subject's
coefficients; the wall has no fluid-coupling h. Mixed-material contact-pair
averaging is not introduced by this file-format change.

`Solv.h` remains the common hydrodynamic smoothing length. `DEM.h` determines
each DEM interpolation and its reaction weights, both with support 2*h.
The particle h/radius arrays are fixed during a run; search maxima are cached.
For heterogeneous DEM h, SPH porosity sums each solid neighbour's contribution
normalized by a fluid-only kernel sum at that SPH point using the neighbour's h.
This reduces to the existing uniform-h normalization when all DEM h agree.

Malformed rows, duplicate/unknown/missing IDs, nonfinite/out-of-range values,
invalid masses/radii/materials and invalid mirror IDs are errors. Header-only
boundary files are allowed for WCSPH; fluid files and enabled DEM must be
nonempty. EISPH needs fluid and ghost points, uniform rho=rho0, and y=0.
These are initial-condition files, not checkpoints: pressures and old contact
history cannot be restored through this format.

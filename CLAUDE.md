# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a scientific research codebase for measuring and modeling **Differential Chromatic Refraction (DCR)** effects in LSST/Rubin Observatory survey data. DCR causes wavelength-dependent positional and shape distortions of astronomical sources due to atmospheric refraction. The project:

- Develops MAF (Metrics Analysis Framework) stackers for DCR first and second moments
- Estimates in-band mean wavelengths from measured DCR signatures
- Analyzes shape distortions across coordinate frames (camera XY → Alt/Az, RA/Dec)
- Validates methodology against LSST DP1 (Data Preview 1) catalogs

Primary development is in Jupyter notebooks; `dcr_interface/` provides a web-based interactive visualization tool on top.

## Running the Web Interface

```bash
cd dcr_interface
pip install -r requirements.txt
bash start_dcr_interface.sh
# or: python dcr_backend.py
# Opens at http://localhost:5000
```

The backend serves a Flask REST API; the frontend is a single HTML file using Plotly.js (`dcr_plotting_interface.html`).

## Key Modules

### Root-level utilities

- **`utils.py`** — Core SED and DCR analysis: `apply_filter()`, `apply_DCR()` (uses GalSim), `generate_templates()`, `estimate_mean_wave()`, `analytic_mean_wave()`, `weighted_avg_and_std()`. This is the primary analysis library used by notebooks.

- **`dcr_utils.py`** — Alternate DCR utilities by the original author (Lugatiman): filter FWHM calculation, refraction/seeing effects. Partially overlaps with `utils.py`.

- **`wavelength_estimate.py`** — Wavelength recovery from DCR measurements: template generation/interpolation, DCR-based inference. Functions are imported directly into analysis notebooks.

- **`rotateMoments.py`** — Coordinate rotations for shape moments: camera XY ↔ Alt/Az ↔ RA/Dec. **Requires the LSST stack** (`lsst.geom`, `lsst.afw`); only usable in a rubin-env environment.

- **`APIaccess.py`** — Thin wrapper for TAP service queries against LSST catalogs.

### `dcr_interface/` package

- **`dcr_backend.py`** — Flask app and `DCRDataProcessor` class. Queries DP1 via TAP or Butler, applies coordinate transforms and DCR calculations, exposes REST endpoints. Can generate mock data for development without LSST access.

- **`coordinate_transforms.py`** — Standalone (no LSST stack) coordinate transform library: shape rotation to RA/Dec and Alt/Az frames, position offset transforms. This is a self-contained version of `rotateMoments.py` for use in the web interface.

- **`demo_custom_backend.py`** — Shows how to subclass `DCRDataProcessor` to inject custom notebook-style calculations.

## Data Dependencies

Notebooks and scripts rely on local data files that are **not in the repo**:

| Path | Contents |
|------|----------|
| `filter_files/total_{u,g,r,i,z,y}.dat` | LSST filter throughputs (wavelength vs transmission) |
| `filter_files/noleak_{u,g,r,i,z,y}.dat` | Filter throughputs without red leaks |
| `1000SEDs.txt` | 1000 galaxy SEDs, r < 26 mag |
| `starSEDs/` | ~4000 stellar SEDs (Kurucz + phoSimMLT) |
| `COSMOS_SED/` | COSMOS SED templates |
| `db files/*.db` | OpSim simulation databases (used for visit scheduling analysis) |
| `DP1_tests/` | Exported DP1 catalog CSVs/VOTables for validation |

## LSST Environment

Several modules require the LSST Science Pipelines stack (`rubin-env`). When running outside the stack:
- Use `dcr_interface/coordinate_transforms.py` instead of `rotateMoments.py`
- `dcr_backend.py` has a `_generate_mock_data()` fallback when Butler/TAP are unavailable
- GalSim must be installed separately: `pip install galsim`

## Notebook Conventions

Primary analysis notebooks:
- `SEDAnalysis.ipynb` — SED propagation, DCR distributions, filter throughput analysis
- `DCRSecondStacker.ipynb` — MAF stacker development for second-moment DCR
- `DCR_in_DP1.ipynb` — First-moment DCR on real DP1 objects
- `wavelength_estimate.ipynb` — Wavelength recovery methodology

Notebooks in `scrap/` and files prefixed with `scrap` are exploratory/deprecated.

## No Formal Test Suite

There is no pytest or unit test framework. Validation is done by:
1. Comparing notebook outputs against DP1 catalog data in `DP1_tests/`
2. Running the web interface with `_generate_mock_data()` for sanity checks
3. Cross-matching against Vizier catalogs (`.vot` files in `DP1_tests/`)

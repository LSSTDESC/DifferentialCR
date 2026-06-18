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

### `PSFs.ipynb` — PSF convolution and FWHM combining formula validation

**Goal:** Verify the OpSim empirical formula for combining atmospheric and system PSF FWHMs, as derived in Xin, Angeli & Ivezić (2016), `papers/Document-20160(1).pdf`.

**Background:** The paper shows that convolving a VonKarman atmospheric PSF with a Gaussian system PSF does *not* yield a total FWHM equal to the naive quadrature sum `sqrt(FWHMatm² + FWHMsys²)`. The correct empirical fits are (Eqs 4.3–4.4):

```
FWHMeff = 1.16 * sqrt(FWHMsys² + 1.04 * FWHMatm²)
FWHMtot = 0.822 * FWHMeff + 0.052
```

**Both PSF components degrade with airmass X = sec(z):**
- `r0(λ, X) = r0(500nm, X=1) * (λ/500)^(6/5) * X^(-3/5)` — Fried parameter (Eq 3.2)
- `FWHMsys(X) = FWHMsys0 * X^(3/5)` — system FWHM (Eq 3.6)
- `FWHMatm` is computed from the scaled r0 via the outer-scale-corrected formula (Eq 3.3)

**LSST reference parameters used:**
- `lam = 500 nm`, `r0 = 0.15 m` at zenith, `L0 = 30 m` (LSST site outer scale)
- `FWHMsys0 = sqrt(0.30² + 0.25² + 0.08²) = 0.4"` (camera + telescope + design in quadrature)

**Notebook structure:**
1. Imports (`galsim`, `numpy`, `matplotlib`)
2. Setup: baseline parameters, airmass `X` (default 1.2), scaled r0 and FWHMsys, `fwhm_atm_analytic()` function (Eq 3.3), GalSim PSF objects
3. FWHM measurement: two functions plus a side-by-side comparison table:
   - `measure_fwhm()` — radial profile interpolation to the half-maximum; precision limited to ~one pixel (0.02")
   - `measure_fwhm_gauss_fit()` — fits a pixelized 2D Gaussian using `scipy.optimize.curve_fit`; the model integrates the continuous Gaussian over each pixel via `erf` (matching GalSim's `drawImage`), so the recovered σ is free of pixelization bias (sub-pixel precision). Fit is restricted to ±2 FWHM of the peak to avoid VonKarman wing contamination. Free parameters: amplitude, sub-pixel centroid (x0, y0), σ.
   - `measure_fwhm_hsm()` — uses `galsim.hsm.FindAdaptiveMom()` (Hirata-Seljak-Mandelbaum adaptive moments); converts `moments_sigma` (pixels) to FWHM in arcsec. Returns `(fwhm, e1, e2)` — the e1/e2 distortion-convention ellipticity components serve as a sanity check (should be ~0 for circularly symmetric PSFs).
4. Wavelength sweep: iterates over wavelengths, measuring GalSim FWHMtot vs Doc-20160 prediction as a function of FWHMatm.
5. Image plot: 1×3 log-scale images of atmospheric, system, and convolved PSFs.
6. Airmass sweep: X from 1.0–2.0, analytic curves for FWHMatm/FWHMsys/FWHMtot(paper)/FWHMtot(quad), with GalSim validation points.

### Section 2 — DCR (cells added below the airmass sweep)

Uses `galsim.ChromaticAtmosphere` to add wavelength-dependent DCR and seeing to the atmospheric PSF, then convolves with the Gaussian system PSF and integrates over a flat SED and a g-band top-hat bandpass (400–550 nm). DCR is oriented along the +y axis (`parallactic_angle=0`), producing y-elongation visible as **negative e1** in HSM.

**Key parameters:** `lam_ref = sys_eff_wavelengths['g']` (~480.7 nm), `alpha = -0.2` (FWHM ∝ λ^α). `lam` and `r0_scaled` are explicitly restored at the top of the section because they are overwritten in the wavelength/airmass sweep loops above.

**Section structure:**
- Setup + draw: builds bandpass/SED/`ChromaticAtmosphere`, draws the three PSF images via `(sed * profile).drawImage(bandpass, ...)`
- FWHM table: image-based helpers `fwhm_radial_img(img)` and `fwhm_hsm_img(img)` (read `img.scale` directly); compares DCR PSF to an achromatic reference at the same wavelength and airmass; reports DCR broadening and e1/e2 shape
- Plot: 1×3 log-scale images (chromatic atm PSF, system PSF, total PSF with DCR) with a shared colorbar. Color scale is controlled by `vmin_frac` and `vmax_frac` at the top of the cell (fractions of the global peak); all three panels share the same `LogNorm` so the colorbar is consistent. Suptitle reports net DCR broadening from both methods.

## No Formal Test Suite

There is no pytest or unit test framework. Validation is done by:
1. Comparing notebook outputs against DP1 catalog data in `DP1_tests/`
2. Running the web interface with `_generate_mock_data()` for sanity checks
3. Cross-matching against Vizier catalogs (`.vot` files in `DP1_tests/`)

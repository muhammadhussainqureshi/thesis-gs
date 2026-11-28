# Gaussian Splatting for Physical Inverse Problems

MS Thesis I, Faculty of Sciences, Information Technology University, Lahore.
Muhammad Hussain Qureshi (MSDS25010); advisor Dr. Ali Murtaza.

3D and 4D Gaussian representations for scientific inverse problems from
[InverseBench](https://github.com/devzhk/InverseBench) (Zheng et al., ICLR 2025):

- **Inverse scattering (2D and 3D):** recover a cell's permittivity from scattered-field measurements with Gaussians
  rendered on a grid and the physics operator in place of a camera. 2D: InverseBench cells. 3D: synthetic nuclei and
  BBBC024 phantom nuclei, with a slice-wise model and a true 3D Born model (sensors on a sphere).
- **Flow reconstruction from sparse observations:** 2D vorticity fields over time (generated with the Navier-Stokes
  solver, Re = 200) reconstructed from all or 10% of the pixels with canonical Gaussians and a 4DGS-style deformation
  model (MLP, GRU or Mamba), compared with a non-learned interpolation baseline.

## Thesis I deliverables
| Item | Where |
|---|---|
| Thesis document (LaTeX + PDF) | [`docs/thesis/`](docs/thesis/) (`main.pdf`) |
| Step-1 report: Mamba vs MLP vs GRU (flow) | [`docs/step1_report/`](docs/step1_report/) (`report.pdf`) |
| Progress report and meeting briefs | [`docs/progress_report/`](docs/progress_report/), [`docs/meeting_brief/`](docs/meeting_brief/), [`docs/mamba_brief/`](docs/mamba_brief/), [`docs/mamba3d_plan/`](docs/mamba3d_plan/) |
| Code | [`src/thesis_gs/`](src/thesis_gs/), [`bin/`](bin/) |
| Experiment configs and Kaggle notebooks | [`configs/`](configs/), [`notebooks/`](notebooks/) |
| Every result (per sample, logs, figures) | [`results/`](results/) (index in [`results/README.md`](results/README.md)) |
| Tests | [`tests/`](tests/) |

## Main findings
**Inverse scattering.** Unbounded Gaussians lose to a regularised voxel grid almost everywhere. Rendering a bounded
density 1 − exp(−Σ) and initialising from the grid makes them beat a *sparsity-regularised* grid under the true 3D model
on every volume with about 9× fewer parameters. A grid with a tuned *total-variation* prior is more accurate still:

| True 3D Born, 10 BBBC024 test volumes, 60 receivers, SNR 34 dB | PSNR | SSIM |
|---|---|---|
| Voxel grid + sparsity + TV (λ_TV = 0.1, tuned on separate volumes) | **28.82** | **0.974** |
| Bounded grid-initialised Gaussians (≈60k params, 2 seeds) | 26.85 | 0.899 |
| + per-Gaussian MLP correction | 26.61 | 0.900 |
| + PointMamba correction | 26.49 | 0.897 |
| Voxel grid + sparsity (524k params) | 25.69 | 0.867 |

Paired over volumes: Gaussians − sparsity grid +1.16 dB (10/10); Gaussians − TV grid −1.97 dB (0/10);
PointMamba − no network −0.36 dB (0/10). A per-scene correction network on free Gaussians adds no constraint.
Adaptive densification grows "ghost" objects in the operator's null space unless a sparsity term is used.

**Flow reconstruction.** A deformation model adds about 10 dB over static Gaussians. Controlled comparison
(10% observed, 5 flows × 3 seeds, ≈69k parameters each, smoothness weight tuned per model on validation frames):

| Unseen frames, 10% observed | PSNR |
|---|---|
| Interpolation (cubic in space, linear in time) | **27.36** |
| GRU | 26.38 |
| Mamba | 25.92 |
| MLP (4DGS-style) | 23.50 |

Mamba − MLP +2.42 dB [1.61, 3.23]; Mamba − GRU −0.46 dB [−1.73, 0.80]. The pre-registered criterion for Mamba
(≥ 1 dB over both, CI above 0) is **not met**: the gain comes from temporal regularisation with a recurrent reading of
each trajectory, not from Mamba. Mamba reproduces the energy spectrum best but overfits the observed pixels.

Full numbers: [`results/README.md`](results/README.md) and the thesis.

## Repository structure
```
src/thesis_gs/     Python package
  gaussians.py       2D/3D anisotropic Gaussian field, local-window rendering, bounded density
  physics.py         scattering operators: InverseBench 2D (per slice) and true 3D Born, noise
  volumes.py         synthetic nuclei + CytoPacq/BBBC024 loaders (.ics/.tif/.npy)
  scatter.py         physics fit (densify/prune, sparsity), voxel grid with sparsity / TV, seeds, tuning splits
  gmamba.py ssm.py   PointMamba over Gaussians; Mamba block with parallel scan, Hilbert serialisation
  ns_data.py         flow sequences from the InverseBench Navier-Stokes solver
  deform.py          deformation networks: MLP, GRU, Mamba (causal / bidirectional / space-time), static
  ns4d.py            flow training and evaluation (validation split, seeds, error split, spectra, interpolation)
  metrics.py utils.py cli.py
bin/               thesis-gs (CLI), run_experiments.py (runs a config, one log per run),
                   ns4d_step1_report.py, scatter3d_report.py (reports with paired CIs)
configs/           experiment configs (one JSON per Kaggle session)
notebooks/         Kaggle notebooks; archive/ holds earlier rounds
results/           all runs (see results/README.md)
docs/              thesis, reports and briefs (LaTeX + PDF)
lib/               third-party code and data notes (InverseBench is cloned at run time)
tests/             unit tests (pytest)
```

## Quick start
```bash
pip install -r requirements.txt
python bin/thesis-gs --task all --quick --data_dir data --out outputs/smoke     # smoke test
python bin/run_experiments.py configs/thesis1_step1_a.json --out outputs/step1_a --data_dir data
python bin/ns4d_step1_report.py outputs/step1_a --out outputs/step1_report
pytest -q
```
InverseBench and its test data are downloaded automatically; BBBC024 volumes are downloaded by the 3D notebooks.
On Kaggle: open a notebook from [`notebooks/`](notebooks/) (GPU T4, Internet on) and use *Save & Run All*.

## License
MIT (see `LICENSE`). Third-party code and data: see [`lib/README.md`](lib/README.md).

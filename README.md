# Gaussian Splatting for Physical Inverse Problems

MS Thesis I (Faculty of Sciences, ITU) — 3D and 4D Gaussian Splatting applied to scientific inverse problems from
[InverseBench](https://github.com/devzhk/InverseBench) (Zheng et al., ICLR 2025).

- **3D Gaussians for linear inverse scattering:** recover a cell's permittivity from scattered-field measurements
  (20 transmitters; 360 / 180 / 60 receivers), in 2D (InverseBench cells) and 3D (CytoPacq / BBBC024 nuclei).
- **4D Gaussians for Navier-Stokes:** canonical Gaussians plus a deformation network over time reconstruct an evolving
  vorticity field from dense or sparse (10%) observations; deformation by an MLP (4DGS-style) or a **Mamba**
  selective state-space model (causal or bidirectional).

The thesis document (LaTeX) is in [`docs/thesis/`](docs/thesis/).

## Repository structure
```
src/thesis_gs/     Python package
  gaussians.py       2D/3D anisotropic Gaussian field, local-window rendering
  physics.py         inverse-scattering forward operator (InverseBench Born model), per slice
  volumes.py         synthetic nuclei + CytoPacq/BBBC024 loaders (.ics/.tif/.npy)
  scatter.py         direct fit, physics fit (densify/prune, sparsity), pixel/voxel baseline
  ns_data.py         Navier-Stokes sequences from the InverseBench solver
  deform.py          deformation networks: MLP, Mamba, bidirectional Mamba, static
  ns4d.py            4D training and evaluation (held-out time steps)
  metrics.py utils.py cli.py
bin/               thesis-gs (CLI from a checkout), run_experiments.py (runs a config, one log per run)
configs/           experiment configs (first run, final part A/B)
notebooks/         Kaggle/Colab notebooks; archive/ holds the notebooks of earlier rounds
results/           summaries, per-sample results and figures of every run so far (see results/README.md)
docs/thesis/       thesis document (LaTeX)
lib/               third-party code and data notes (InverseBench is cloned at run time)
tests/             unit tests (pytest)
```

## Quick start
```bash
pip install -r requirements.txt
python bin/thesis-gs --task all --quick --data_dir data --out outputs/smoke   # 2-minute smoke test
python bin/thesis-gs --task scatter2d --l1 0.3 --data_dir data --out outputs/scatter2d
python bin/run_experiments.py configs/final_part_a.json --out outputs/final_a --volumes /path/to/bbbc024
pytest -q                                                                      # unit tests
```
InverseBench and its test data are downloaded automatically. On Kaggle use
[`notebooks/kaggle_final_experiments.ipynb`](notebooks/kaggle_final_experiments.ipynb) (GPU, Internet on, *Save & Run All*).

## Main results (PSNR dB / SSIM; final runs Nov 7, 2026)
| Inverse scattering, 2D real cells (5 samples) | 360 receivers | 180 | 60 |
|---|---|---|---|
| Gaussians + sparsity (0.3) | 30.8 / 0.980 | 30.6 / 0.979 | 27.1 / 0.924 |
| Pixel grid | **36.1 / 0.984** | **34.7** / 0.974 | **28.6** / 0.886 |
| Pixel grid + same sparsity | 33.2 / **0.985** | 32.5 / **0.983** | 27.7 / **0.933** |

| Inverse scattering, 3D (3 volumes) | 360 | 180 | 60 |
|---|---|---|---|
| Synthetic nuclei: Gaussians | 40.8 / **0.994** | 40.6 / **0.994** | **40.9 / 0.988** |
| Synthetic nuclei: voxel grid | **44.1** / 0.987 | **41.5** / 0.977 | 37.3 / 0.932 |
| BBBC024 nuclei: Gaussians | 27.9 / 0.967 | 27.6 / 0.965 | 27.0 / 0.943 |
| BBBC024 nuclei: voxel grid | **41.8** / 0.989 | **40.3** / 0.980 | **35.3** / 0.936 |
| BBBC024 nuclei: Gaussians + sparsity (0.3) | 28.4 / 0.974 | 28.4 / 0.974 | 27.4 / 0.955 |
| BBBC024 nuclei: voxel grid + same sparsity | 38.5 / **0.996** | 38.0 / **0.995** | 34.4 / **0.971** |

| True 3D scattering (3D Born, sensors on a sphere), sparsity 0.3 both | 360 | 180 | 60 |
|---|---|---|---|
| Synthetic: Gaussians (≈42k params) | 37.6 / 0.977 | 36.6 / 0.955 | 32.1 / 0.852 |
| Synthetic: voxel grid (524k params) | **42.1 / 0.985** | **39.9 / 0.976** | **38.2 / 0.966** |
| BBBC024: Gaussians (≈47k params) | 25.6 / 0.960 | 24.9 / 0.940 | 20.9 / 0.772 |
| BBBC024: voxel grid (524k params) | **32.5 / 0.983** | **30.5 / 0.956** | **25.6 / 0.875** |

**Bounded Gaussians** (`--bounded`: render 1 − exp(−Σ), in [0, 1) like the clipped grid; `--init_from_grid`), true 3D, same sparsity for both:

| True 3D, bounded + grid-initialised Gaussians vs voxel grid | 360 | 180 | 60 |
|---|---|---|---|
| BBBC024: Gaussians (≈60k params) | **32.8** / 0.990 | **31.4** / 0.980 | **26.7** / 0.909 |
| BBBC024: voxel grid (524k params) | 32.5 / 0.983 | 30.5 / 0.956 | 25.6 / 0.875 |
| Synthetic: Gaussians (≈54k params) | **46.4 / 0.993** | **43.6 / 0.986** | **39.1** / 0.963 |
| Synthetic: voxel grid (524k params) | 42.1 / 0.985 | 39.9 / 0.976 | 38.2 / 0.966 |

Bounded Gaussians win on every volume at every receiver count in true 3D. Slice-wise 3D and 2D (more measurements): the regularised grid is still slightly ahead (see `results/README.md`).

2D on 10 cells (paired, unbounded Gaussians): the pixel grid with the same sparsity beats Gaussians on every cell (by 0.6 / 2.3 / 2.9 dB at 60 / 180 / 360 receivers). Generalised Gaussians (β = 2, 4) did not help under the physics loss.

| Navier-Stokes 4D, held-out frames (PSNR, 3 sequences) | static | MLP | Mamba | bidir. Mamba |
|---|---|---|---|---|
| 100% observed | 17.0 | 27.6 | 27.9 | – |
| 100% observed + temporal smoothness | – | **27.4** | 27.2 | 26.4 |
| 10% observed | 16.0 | 23.2 | 23.1 | – |
| 10% observed + temporal smoothness | – | 23.0 | **25.4** | 25.1 |

Details, per-sample numbers and figures: [`results/README.md`](results/README.md).

## License
MIT (see `LICENSE`). Third-party code and data: see [`lib/README.md`](lib/README.md).

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

## Main results so far (mean of 3 samples, PSNR dB / SSIM)
| Inverse scattering, 2D real cells | 360 receivers | 180 | 60 |
|---|---|---|---|
| Gaussians | 30.7 / 0.959 | 30.4 / 0.957 | 26.5 / 0.814 |
| Gaussians + sparsity (0.3) | 31.4 / 0.983 | 31.0 / 0.982 | 28.2 / **0.940** |
| Pixel grid | **36.7 / 0.985** | **35.3 / 0.976** | **29.2** / 0.878 |

| Inverse scattering, 3D synthetic nuclei | 360 | 180 | 60 |
|---|---|---|---|
| Gaussians | 40.8 / 0.994 | 40.6 / 0.994 | **40.9 / 0.988** |
| Voxel grid | **44.1** / 0.987 | **41.5** / 0.977 | 37.3 / 0.932 |

| Navier-Stokes 4D, held-out frames (PSNR) | static | MLP | Mamba |
|---|---|---|---|
| 100% observed | 17.0 | 27.6 | 27.9 |
| 10% observed | 16.0 | 23.2 | 23.1 |

Details, per-sample numbers and figures: [`results/README.md`](results/README.md).

## License
MIT (see `LICENSE`). Third-party code and data: see [`lib/README.md`](lib/README.md).

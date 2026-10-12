# Gaussian Splatting for Physical Inverse Problems

Thesis code: 3D Gaussian Splatting on **linear inverse scattering** and 4D Gaussian Splatting on
**Navier-Stokes**, using datasets from InverseBench (Zheng et al., ICLR 2025). Later: Mamba.

Full project context and plan: [`docs/thesis_project_context.md`](docs/thesis_project_context.md)

## How we work
- **Code lives here on GitHub.** Edit on github.com or locally, commit, push.
- **Everything runs on Google Colab (GPU).** Notebooks in `notebooks/` clone/pull this repo.
- **Data and results live on Google Drive** (`MyDrive/thesis/`), so they survive Colab restarts.

## Layout
```
day1/        check_env.py, load_scatter_data.py
notebooks/   00_setup_colab.ipynb   (Day 1 runner)
docs/        thesis_project_context.md
```

## Start
Open `notebooks/00_setup_colab.ipynb` in Colab:
`https://colab.research.google.com/github/<USER>/<REPO>/blob/main/notebooks/00_setup_colab.ipynb`

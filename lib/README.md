# Third-party code

| Dependency | Use | How it is obtained | License |
|---|---|---|---|
| [InverseBench](https://github.com/devzhk/InverseBench) (Zheng et al., ICLR 2025) | Scattering forward operator (`inverse_problems/inverse_scatter.py`) and Navier-Stokes solver (`inverse_problems/navier_stokes.py`) | Cloned automatically at run time into `--ib_dir` (default `/tmp/InverseBench`); not vendored | MIT |
| InverseBench datasets (CaltechDATA) | `inv-scatter-test`, `navier-stokes-test` | Downloaded automatically into `--data_dir` | see the dataset card |
| [BBBC024](https://bbbc.broadinstitute.org/BBBC024) (Broad Bioimage Benchmark Collection) | Real 3D HL60 nucleus masks generated with CytoPacq | Downloaded by the Kaggle notebook | see the BBBC page |

All other dependencies are Python packages listed in `requirements.txt`.

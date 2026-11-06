# Results

Each folder holds `summary.csv` (means), `results.csv` (per sample), `reconstructions.png` and `run_config.json`
(exact settings). All runs: Kaggle/Colab T4 GPU, 3000 iterations, measurement noise σ = 1e-4.

| Folder | What | Key outcome |
|---|---|---|
| `2026-10-29_first_run/` | All tasks, default settings; 3D on synthetic nuclei | 2D: pixels win; 3D: Gaussians win at 60 receivers (+3.6 dB); 4D: deformation +10 dB, Mamba ≈ MLP on held-out frames |
| `2026-10-29_round1_more_gaussians/` | 2D with ~2000 Gaussians (aggressive densification) | Worse (≈ 23 dB): ghost copy of the cell in the measurements' blind region |
| `2026-10-30_round2_sparsity/` | 2D with sparsity weight 0.3 / 1 / 3 | 0.3 removes ghosts; at 60 receivers SSIM 0.940 vs pixels 0.878 |
| *(pending)* final part A / B | fair pixel baseline with sparsity, 5 samples; 3D on real BBBC024 cells; 4D with temporal smoothness and bidirectional Mamba | — |

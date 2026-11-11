# Results

Each folder holds `summary.csv` (means), `results.csv` (per sample), `reconstructions.png` and `run_config.json`
(exact settings). All runs: Kaggle/Colab T4 GPU, 3000 iterations, measurement noise σ = 1e-4.

| Folder | What | Key outcome |
|---|---|---|
| `2026-10-29_first_run/` | All tasks, default settings; 3D on synthetic nuclei | 2D: pixels win; 3D: Gaussians win at 60 receivers (+3.6 dB); 4D: deformation +10 dB, Mamba ≈ MLP on held-out frames |
| `2026-10-29_round1_more_gaussians/` | 2D with ~2000 Gaussians (aggressive densification) | Worse (≈ 23 dB): ghost copy of the cell in the measurements' blind region |
| `2026-10-30_round2_sparsity/` | 2D with sparsity weight 0.3 / 1 / 3 | 0.3 removes ghosts; at 60 receivers SSIM 0.940 vs pixels 0.878 |
| `2026-11-07_final/part_a/` | 2D, 5 samples, sparsity 0.1 / 0.3, and pixel grid with the same sparsity (`_fair`); 3D on real BBBC024 nuclei, sparsity 0 / 0.3 | 2D: SSIM gain at 60 rec. holds (0.924 vs 0.886) but vanishes when the grid gets the same sparsity (0.933); 3D BBBC024: voxel grid far better (34–42 vs 27–29 dB), Gaussians blur sharp mask edges |
| `2026-11-10_bounded/` | Bounded density 1 − exp(−Σ) and grid-initialised Gaussians: true 3D (BBBC024, synthetic), slice-wise BBBC024, 2D on 10 cells | True 3D: Gaussians beat the voxel grid on every volume (BBBC024 +0.3 / +1.0 / +1.1 dB, synthetic +4.3 / +3.7 / +0.9 dB at 360 / 180 / 60 rec.) with ~9× fewer params; slice-wise BBBC024 35.8 / 35.7 / 34.1 vs grid 38.5 / 38.0 / 34.4; 2D: +0.4 dB at 60 rec., −0.2 to −1.4 dB at 180 / 360 |
| `2026-11-08_extra/` | True 3D Born scattering (born3d_*), generalised Gaussians β = 4 (3D) and β = 2 (2D), 2D on 10 cells with std | True 3D: voxel grid better everywhere (BBBC024 32.5 vs 25.6 dB at 360 rec.), Gaussians 11–12× more compact; β > 1 worse under physics; 10-cell paired: grid + sparsity wins on every cell |
| `2026-11-07_final/part_b/` | 4D with temporal smoothness (λ_T = 1): MLP, Mamba, bidirectional Mamba | 10% observations: Mamba 25.4 dB vs MLP 23.0 dB held-out (better on all 3 sequences); 100%: equal (27.2 vs 27.4); bidirectional no gain, 2× slower |

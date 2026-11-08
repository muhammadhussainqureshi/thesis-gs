# Results

Each folder holds `summary.csv` (means), `results.csv` (per sample), `reconstructions.png` and `run_config.json`
(exact settings). All runs: Kaggle/Colab T4 GPU, 3000 iterations, measurement noise σ = 1e-4.

| Folder | What | Key outcome |
|---|---|---|
| `2026-10-29_first_run/` | All tasks, default settings; 3D on synthetic nuclei | 2D: pixels win; 3D: Gaussians win at 60 receivers (+3.6 dB); 4D: deformation +10 dB, Mamba ≈ MLP on held-out frames |
| `2026-10-29_round1_more_gaussians/` | 2D with ~2000 Gaussians (aggressive densification) | Worse (≈ 23 dB): ghost copy of the cell in the measurements' blind region |
| `2026-10-30_round2_sparsity/` | 2D with sparsity weight 0.3 / 1 / 3 | 0.3 removes ghosts; at 60 receivers SSIM 0.940 vs pixels 0.878 |
| `2026-11-07_final/part_a/` | 2D, 5 samples, sparsity 0.1 / 0.3, and pixel grid with the same sparsity (`_fair`); 3D on real BBBC024 nuclei, sparsity 0 / 0.3 | 2D: SSIM gain at 60 rec. holds (0.924 vs 0.886) but vanishes when the grid gets the same sparsity (0.933); 3D BBBC024: voxel grid far better (34–42 vs 27–29 dB), Gaussians blur sharp mask edges |
| `2026-11-07_final/part_b/` | 4D with temporal smoothness (λ_T = 1): MLP, Mamba, bidirectional Mamba | 10% observations: Mamba 25.4 dB vs MLP 23.0 dB held-out (better on all 3 sequences); 100%: equal (27.2 vs 27.4); bidirectional no gain, 2× slower |

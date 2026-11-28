# 3D controls: true 3D Born, BBBC024, 60 receivers

Test volumes 2-11 (volumes 0-1 only for tuning the TV weight). Noise: complex Gaussian on the measured field, sigma = 2% (SNR 34 dB) or 10% (SNR 20 dB) of rms(y).

## SNR 34 dB: TV weight on tuning volumes 0-1 (PSNR dB; sparsity-only grid 25.74)

|   tv |   PSNR |
|-----:|-------:|
| 0.01 |  27.37 |
| 0.03 |  28.12 |
| 0.1  |  28.39 |
| 0.3  |  27.59 |
| 1    |  25.33 |
| 3    |  21.71 |

Chosen: 0.1

## SNR 34 dB: test volumes

| method                           |   volumes |   seeds |   PSNR |   PSNR SD (volumes) |   SSIM |
|:---------------------------------|----------:|--------:|-------:|--------------------:|-------:|
| + per-Gaussian MLP               |        10 |       1 | 26.611 |               1.936 |  0.9   |
| Gaussians (B+G), no network      |        10 |       2 | 26.849 |               1.768 |  0.899 |
| + PointMamba                     |        10 |       2 | 26.486 |               1.859 |  0.897 |
| Voxel grid + sparsity            |        10 |       1 | 25.694 |               2.002 |  0.867 |
| Voxel grid + sparsity + TV (0.1) |        10 |       1 | 28.819 |               1.316 |  0.974 |


Paired differences (per volume, seed-averaged):

| comparison                                   |   mean dB | 95% CI         | first better   |
|:---------------------------------------------|----------:|:---------------|:---------------|
| per-Gaussian MLP - no network                |     -0.24 | [-0.42, -0.06] | 1/10           |
| PointMamba - no network                      |     -0.36 | [-0.45, -0.28] | 0/10           |
| Gaussians - Voxel grid + sparsity            |      1.16 | [+0.94, +1.37] | 10/10          |
| Gaussians - Voxel grid + sparsity + TV (0.1) |     -1.97 | [-2.35, -1.59] | 0/10           |


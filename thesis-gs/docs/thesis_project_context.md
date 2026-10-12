# Research Thesis — Project Context & Action Plan

> Living document. Purpose: give full context to anyone (including Claude in a future chat) picking this project up later. Update the **Status Log** at the bottom as work progresses.

- **Started this phase:** Sunday, October 11, 2026
- **Current focus:** Linear inverse scattering (3D Gaussian Splatting) → then Navier-Stokes (4D Gaussian Splatting) → then Mamba
- **Current deadline:** 2-week sprint for inverse scattering, Oct 11 – Oct 24, 2026

---

## 1. Thesis background — where we started

### 1.1 Original thesis direction
The thesis is built around **4D LangSplat (Li et al., CVPR 2025)** — a 4D (3D + time) language field: a dynamic scene that answers both time-agnostic queries ("where is the cup?") and time-sensitive queries ("when is coffee poured?"). The stated thesis plan from the first presentation: **reproduce 4D LangSplat, find its failure cases, and improve one component.**

### 1.2 First presentation (already given to the professor)
File: `4D_Language_Gaussian_Splatting.pptx`. It walks the lineage of methods leading to 4D LangSplat, each slide answering *What is it / Input / Output / How it works / How it connects to the next step*:

| # | Method | Key idea | Link to next step |
|---|---|---|---|
| 2 | **NeRF** | MLP maps 5D input (x,y,z,θ,φ) → color + density; volume rendering along rays; trained with MSE on pixels | Slow, implicit |
| 3 | **3D Gaussian Splatting** (Kerbl et al., 2023) | Scene = millions of 3D Gaussians (position, shape, opacity, color); splat + alpha-blend; real-time; densify/prune | Static scenes only |
| 4 | **D-NeRF** (Pumarola et al., 2021) | Canonical space + deformation network for dynamic scenes | 4DGS reuses this skeleton |
| 5 | **4DGS** (Wu et al., 2024) | Canonical Gaussians + deformation network predicting per-timestamp motion/shape | Geometry has no meaning yet |
| 6 | **CLIP** (Radford et al., 2021) | Joint image–text embedding space | Enables open-vocabulary search |
| 7 | **SAM** (Kirillov et al., 2023) | Class-agnostic object masks | One clean CLIP feature per object |
| 8 | **LangSplat** (Qin et al., 2024) | CLIP features per SAM mask, compressed by autoencoder, stored per Gaussian | Static only |
| 9 | **4D LangSplat** (Li et al., CVPR 2025) | MLLM writes object captions → LLM sentence embeddings as supervision → status deformable network | Thesis starting point |

Note: Slide 1 of that deck contains leftover placeholder text unrelated to the topic (CNN, batch size 32/64, accuracy) — fix before reusing the deck.

---

## 2. New assignment from the professor

The professor gave the paper **InverseBench: Benchmarking Plug-and-Play Diffusion Priors for Inverse Problems in Physical Sciences** (Zheng et al., ICLR 2025, Caltech). **The paper itself is not the research topic** — it was given only as a **source of datasets**. Task: pick the datasets best suited to 3D and 4D Gaussian Splatting and justify the choice.

### 2.1 The five InverseBench datasets (what the images are)
| Dataset | What the images show | Size / resolution |
|---|---|---|
| **Linear inverse scattering** (CytoPacq) | Grayscale simulated cell nuclei (1–6 HL60 nuclei), bright blobs on dark background; pixel = permittivity | 10,000 train, 128×128 |
| **Compressed sensing MRI** (fastMRI knee) | Real grayscale knee MRI slices (bone, cartilage, soft tissue, some injuries) | 25,012 train slices, 320×320, complex |
| **Black hole imaging** (GRMHD) | Simulated glowing orange ring around dark center (like M87*) | 50,000 train, 64×64; training data **not public** |
| **Full waveform inversion** (OpenFWI CurveFault-B) | Colored maps of underground rock layers with faults; color = wave speed | 50,000 train, 128×128 (resized from 70×70) |
| **Navier-Stokes** | Colored 2D fluid vorticity swirls (red/blue = opposite rotation) | 10,000 (paper) / 20,480 (dataset card) train, 128×128 |

### 2.2 Core insight used in the analysis
3DGS/4DGS are designed for **posed RGB photos/videos** of 3D scenes. All InverseBench datasets are **2D physical fields** recovered from **physics measurements** (scattered light, k-space, visibilities, seismic waveforms, sparse field samples). So **none is a drop-in fit**; Gaussians are used here as a **representation** of a volume/field, evaluated on a grid, with a physics forward model instead of a camera renderer.

### 2.3 Ranking we arrived at
1. **CS-MRI** — best practical 3D fit (real 3D volumes; Gaussian's Fourier transform is a Gaussian → k-space friendly). Not chosen by the professor.
2. **Navier-Stokes** — best 4D fit (only dataset that genuinely evolves over time).
3. **Linear inverse scattering** — good 3D fit (smooth blob-like cells; transmitters around the object ≈ cameras around a scene; CytoPacq can produce 3D cells).
- Rejected: **Black hole** (single viewpoint, data not public), **FWI** (static rocks, sharp edges bad for smooth Gaussians, expensive PDE).

### 2.4 Professor's decision
- **3D Gaussian Splatting → Linear inverse scattering**
- **4D Gaussian Splatting → Navier-Stokes**
- Then **test on Mamba**
- **"No benchmarking"** (interpretation: no comparison with InverseBench's diffusion methods; we still report our own metrics — **to confirm**)
- Order chosen by student: **inverse scattering first**, then Navier-Stokes.

---

## 3. Key concepts

### 3.1 Mamba
Mamba (Gu & Dao, 2023) is a sequence model based on selective state space models — like a Transformer in role, but linear-time in sequence length (Transformers are quadratic). Code: https://github.com/state-spaces/mamba (needs NVIDIA GPU + CUDA). Two candidate uses (**ask professor which**):
- **Navier-Stokes:** replace the 4DGS deformation network with a Mamba block over the time sequence (most natural).
- **Both datasets:** treat the list of Gaussians as a token sequence processed by Mamba.

### 3.2 How Gaussians are used here (differs from original 3DGS)
- No cameras, no COLMAP, no splatting to image planes.
- Field value at each pixel/voxel = sum of Gaussians evaluated there (simple PyTorch).
- Loss = compare **forward_model(Gaussian field)** with **measurements** (inverse problem), or compare directly with ground truth (sanity step).
- Keep densification/pruning ideas from 3DGS.
- Store Gaussian parameters as one tensor of shape (N, params) → easy to feed into Mamba later.

---

## 4. Dataset details: what we have vs. what is missing

### 4.1 Linear inverse scattering (3D target)
- **Have:** 2D 128×128 cell images (values 0–1) + the 2D physics forward model (Born approximation, 20 transmitters on a circle, 360/180/60 receivers) in the InverseBench repo.
- **Missing:**
  1. **3D cell volumes** → generate with **CytoPacq** (confirmed: it outputs fully 3D phantoms with configurable z-step, plus reference annotations; HL60 nucleus static/dynamic available; preset BBBC024 = HL60 nucleus, 100×70×16 µm). Download the **phantom** (clean ground truth), not the blurred microscope image. InverseBench used VOI 42×42×12, HL60 nucleus (static), random position, 1–6 nuclei.
  2. **3D forward model** → **Option A (chosen for the 2-week sprint):** apply the existing 2D forward model slice by slice. **Option B (future):** true 3D Born model with 3D Green's function and sensors on a sphere.
- Note: original scattering images are fluorescence-style phantoms treated as permittivity (0–1); do the same for 3D.

### 4.2 Navier-Stokes (4D target)
- **Have:** 2D vorticity fields (128×128, range −10..10, Reynolds number 200, periodic domain) + the Navier-Stokes solver (forward model) in the InverseBench repo.
- **Missing:**
  1. **Time sequences** — downloaded data are single snapshots. Re-run the repo's solver on each sample and **save every time step**.
  2. **2D renderer** — the official 4DGS code assumes 3D scenes + cameras; we just evaluate 2D Gaussians on the pixel grid (simpler).
- **2D + time is acceptable** — same idea as 4DGS (canonical Gaussians + deformation network), one dimension lower.
- Benchmark note: in InverseBench this problem only allows black-box access to the solver (no gradients). If we need gradients through the solver, check feasibility.

### 4.3 Known discrepancies (paper vs. dataset card)
- Navier-Stokes train: paper 10,000 vs. card 20,480
- MRI test: paper 94 vs. card 96
- FWI test: card lists 100, but results reported on 10 (simulation cost)

---

## 5. Links

**InverseBench**
- Data (CaltechDATA): https://data.caltech.edu/records/zg89b-mpv16 (README links a different token-based record; try both)
- Dataset card: https://github.com/devzhk/InverseBench/blob/main/Dataset_Card.md
- Code: https://github.com/devzhk/InverseBench
- Project page: https://devzhk.github.io/InverseBench/
- Pretrained diffusion models: https://github.com/devzhk/InverseBench/releases/tag/diffusion-prior
- Paper: https://arxiv.org/abs/2503.11043

**Original sources**
- CytoPacq simulator: https://cbia.fi.muni.cz/simulator/index.php (info: https://cbia.fi.muni.cz/research/simulations/cytopacq.html)
- fastMRI: https://fastmri.med.nyu.edu/ (agreement required; code https://github.com/facebookresearch/fastMRI); fastMRI+ labels (unverified link): https://github.com/microsoft/fastmri-plus
- OpenFWI: https://openfwi-lanl.github.io/
- GRMHD black hole source paper: https://iopscience.iop.org/article/10.3847/1538-4365/ac582e
- Navier-Stokes: generated by InverseBench authors; no external source

**Tools / references**
- gsplat (nerfstudio) — 3DGS library
- 4DGS official code: hustvl/4DGaussians
- GaussianImage (ECCV 2024) — reference for 2D Gaussian image fitting
- Mamba: https://github.com/state-spaces/mamba
- Related Gaussian/neural work for scientific reconstruction (verify before citing): R²-Gaussian (CT, 2024), X-Gaussian (X-ray, 2024), BH-NeRF (Levis et al., 2022)

---

## 6. Action plan — Linear inverse scattering (2-week sprint, Oct 11 – Oct 24, 2026)

Cut to fit 2 weeks: Option A physics only; experiments limited to receivers + Gaussian count; small sample counts.

### Week 1 — 2D (make the method work)
| Date | Task | Done when |
|---|---|---|
| **Sun Oct 11** | Set up Python + PyTorch on a GPU; clone InverseBench; download CaltechDATA. **Submit CytoPacq jobs** (HL60 static, few nuclei, ~64×64×32, 10 volumes). Ask professor to approve Option A. | Can load & show a 2D cell image |
| **Mon Oct 12** | Load/visualize 2D images. Download CytoPacq phantoms → NumPy, normalize 0–1, view slices. | 3D slices displayed |
| **Tue Oct 13 – Wed Oct 14** | Fit N 2D Gaussians directly to an image; add densification + pruning. | PSNR > 30 dB on 5 images |
| **Thu Oct 15 – Fri Oct 16** | 2D Gaussians through physics: loss on A·(Gaussian image) vs. measurements y. Run 360 → 180 → 60 receivers. | Reconstructions + PSNR/SSIM |
| **Sat Oct 17** | Buffer; send professor 2D update (images + table). | Update sent |

### Week 2 — 3D (the goal)
| Date | Task | Done when |
|---|---|---|
| **Sun Oct 18 – Mon Oct 19** | 3D Gaussians (3D position, 3 scales, rotation, density) fitted directly to 3–5 CytoPacq volumes; evaluate only within ~3σ of each center. | Good PSNR/SSIM, slices match |
| **Tue Oct 20 – Wed Oct 21** | 3D through physics, Option A (2D forward model per slice). | 3D reconstruction from measurements |
| **Thu Oct 22** | Experiments: receivers 360/180/60; vary #Gaussians. | Metrics recorded |
| **Fri Oct 23** | Figures (slices vs. ground truth, quality vs. receivers) + summary table. | Figures ready |
| **Sat Oct 24** | Slides (same "What/Input/Output/How/Connection" style) + buffer. | Presentation ready |

**Metrics to report:** PSNR, SSIM, relative L2 error, measurement error, number of Gaussians, training time.

**Risks & fallbacks**
- CytoPacq delay → continue 2D; swap in volumes when ready.
- No GPU → 2D works on CPU; 3D needs GPU — confirm access on day 1.
- 3D physics not working by Wed Oct 21 → present 3D direct fit + 2D physics results.

---

## 7. Next phases (after inverse scattering)

### 7.1 Navier-Stokes — 4DGS (2D + time)
1. Generate time sequences with the InverseBench solver (save every step).
2. Fit canonical 2D Gaussians to the first time step.
3. Add a deformation network: (Gaussian, t) → change in position, shape, value (4DGS design).
4. Train on whole sequences.
5. Test: held-out time steps; reconstruction from sparse observations (inverse problem).

### 7.2 Mamba
- Navier-Stokes: replace the deformation network with Mamba over time; compare plain 4DGS vs. 4DGS + Mamba with everything else identical.
- Scattering: Gaussians-as-tokens, if the professor wants it.
- Core result: **plain GS vs. GS + Mamba**.

---

## 8. Open questions for the professor
1. 3D physics: Option A (slice-wise 2D forward model) acceptable, or full 3D scattering model required?
2. Is 2D + time acceptable for Navier-Stokes 4D?
3. Where should Mamba go: deformation network replacement, or Gaussians as tokens?
4. "No benchmarking" = no comparison with InverseBench methods, but we still report our own metrics?
5. How does this work relate back to 4D LangSplat in the thesis (side project vs. new main direction)?

---

## 9. Code & technical notes (learned while building)

### 9.1 Where things live (decided Oct 12)
- **Compute: Google Colab (T4 GPU).** Mac not used for running code (Apple GPU lacks complex128; Mamba needs CUDA).
- **Code: GitHub repo `thesis-gs`** (student's account). Colab notebooks clone/pull it each run.
- **Data + results: Google Drive `MyDrive/thesis/`** (`data/*.zip` cached once, `outputs/<day>/`).
  Each session unzips data to Colab local disk `/content/data/` (LMDB is faster off local disk).
- InverseBench is cloned fresh into `/content/InverseBench` each session (not committed).
```
thesis-gs/ (GitHub)
├── day1/        check_env.py, load_scatter_data.py
├── notebooks/   00_setup_colab.ipynb  (Day 1 runner: GPU check, Drive, clone, data, run)
└── docs/        thesis_project_context.md (this file)
```
- CaltechDATA direct file URL pattern: `https://data.caltech.edu/records/zg89b-mpv16/files/<name>.zip?download=1`
  Files: inv-scatter-{train 84MB, val 100KB, test 840KB}, navier-stokes-{train 1.2GB, val, test 6MB}, knee MRI ~32GB (not needed).
- Private repo in Colab: add `GITHUB_TOKEN` in Colab Secrets.

### 9.2 Inverse scattering data & forward model (from the repo code)
- Data = LMDB; key `"0"`, `"1"`, ... → raw float32 bytes, reshape to (1,128,128), values 0..1.
  Benchmark loader normalizes with mean 0.5, std 0.25 → (x-0.5)/0.5 → range [-1,1].
- Forward model: `inverse_problems/inverse_scatter.py`, class `InverseScatter`
  (Lx=Ly=0.18 m, 128×128, wave=6, numTrans=20, numRec=360/180/60, sensorRadius=1.6, sigma_noise=1e-4,
  unnorm_shift=1.0, unnorm_scale=0.5). Build with `svd=False` (SVD takes 10–20 min, not needed).
- `forward(f, unnormalize=True)` expects [-1,1] input and maps back via (x+1)*0.5.
  **For Gaussians we work in raw 0..1 and call `forward(f, unnormalize=False)`** — verified identical (diff ~1e-10).
- **`forward()` only uses the first item in a batch** (`f[0,0]`) → always pass one image at a time.
- Output: complex128, shape (1, 20, numRec). Magnitudes ~1e-2 → scale the data loss accordingly.
- Loss in repo: sum |A f − y|². Fully differentiable in PyTorch → can backprop to Gaussian params.
- Forward model builds in ~2 s on CPU; one forward pass is a single matrix multiply (fast).

## 10. Status log
| Date | Update |
|---|---|
| 2026-10-11 | Project context written. Professor assigned inverse scattering (3D) and Navier-Stokes (4D), then Mamba. Starting 2-week inverse scattering sprint today. |
| 2026-10-11 | Day 1: cloned InverseBench, read its scattering code (notes in §9). Wrote `day1/check_env.py` + `day1/load_scatter_data.py` (load LMDB, plot cells, run forward model); tested on synthetic stand-in data. Student to: run on GPU machine with real CaltechDATA, submit CytoPacq jobs, send Option A question to professor. |
| 2026-10-12 | Switched to **Colab + GitHub + Google Drive** workflow (§9.1). Built repo `thesis-gs` with `notebooks/00_setup_colab.ipynb`; data-unzip logic and loader tested on stand-in data. Student to: create GitHub repo, push, run notebook on Colab T4. CytoPacq jobs + professor email still pending. |

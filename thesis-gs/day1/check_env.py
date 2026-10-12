"""Day 1 - Step 1: check that Python, PyTorch and the hardware are ready.

Run:  python check_env.py
"""
import platform
import sys

print(f"Python  : {sys.version.split()[0]}  ({platform.system()} {platform.machine()})")

try:
    import torch
except ImportError:
    sys.exit("PyTorch is not installed -> pip install torch")

print(f"PyTorch : {torch.__version__}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"GPU {i}   : {p.name}, {p.total_memory / 1024**3:.1f} GB (CUDA)")
elif torch.backends.mps.is_available():
    x = torch.randn(1000, 1000, device="mps")
    ok = bool((x @ x).isfinite().all())
    print(f"GPU     : Apple Silicon (MPS) - test {'OK' if ok else 'FAILED'}")
    print("          Note: MPS has no float64/complex128, so the scattering physics runs on CPU.")
else:
    print("GPU     : none (CPU only). Fine for the 2D steps.")

# The scattering physics needs complex128 on CPU - check it works.
a = torch.randn(256, 256, dtype=torch.complex128)
print(f"CPU complex128 test: {'OK' if (a @ a).isfinite().all() else 'FAILED'}")

missing = []
for pkg in ["numpy", "scipy", "matplotlib", "lmdb"]:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)
print("Packages: all present" if not missing else f"Packages: missing {missing} -> pip install {' '.join(missing)}")

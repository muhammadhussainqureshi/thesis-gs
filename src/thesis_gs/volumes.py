"""3D cell volumes: synthetic HL60-like nuclei and loaders for CytoPacq / BBBC024 files (.ics/.tif/.npy)."""
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F

from .utils import log, pip_install


def synthetic_nuclei_volume(D, H, W, seed):
    """Stand-in for CytoPacq HL60 nuclei: 1-6 soft rotated ellipsoids with smooth internal texture, 0..1."""
    rng = np.random.default_rng(seed)
    z, y, x = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing="ij")
    vol = np.zeros((D, H, W), np.float32)
    for _ in range(rng.integers(1, 7)):
        c = np.array([rng.uniform(0.35, 0.65) * D, rng.uniform(0.25, 0.75) * H, rng.uniform(0.25, 0.75) * W])
        r = np.array([rng.uniform(0.15, 0.25) * D, rng.uniform(8, 16), rng.uniform(8, 16)])
        a = rng.uniform(0, np.pi)
        yy, xx = y - c[1], x - c[2]
        yr, xr = np.cos(a) * yy - np.sin(a) * xx, np.sin(a) * yy + np.cos(a) * xx
        dist = np.sqrt(((z - c[0]) / r[0]) ** 2 + (yr / r[1]) ** 2 + (xr / r[2]) ** 2)
        shape = 1 / (1 + np.exp((dist - 1) * 12))
        k = rng.normal(size=(3, 3))
        tex = 1 - 0.25 * (np.sin(k[0, 0] * z / 3 + k[0, 1] * y / 5 + k[0, 2] * x / 5) *
                          np.cos(k[1, 0] * z / 4 + k[1, 1] * y / 6 + k[1, 2] * x / 6)) ** 2
        vol = np.maximum(vol, shape * tex * rng.uniform(0.6, 1.0))
    return np.clip(vol, 0, 1).astype(np.float32)


def read_ics(path):
    """Minimal reader for ICS microscopy files (v1: header .ics + raw .ids; v2: data right after the
    'end' line, as in BBBC024). Handles LF / CRLF / CR line endings and gzip/zlib compression.
    Returns an array in (z, y, x) order."""
    import gzip
    import re
    import zlib
    raw = open(path, "rb").read()
    m = re.search(rb"(?:\r\n|\n|\r)end[ \t]*(?:\r\n|\n|\r)", raw)   # header ends with an "end" line
    header = raw[:m.start() if m else len(raw)].decode("latin-1")
    info = {}
    for line in re.split(r"\r\n|\n|\r", header):
        parts = [x for x in line.strip().split("\t") if x != ""]
        if len(parts) >= 3:
            info[(parts[0].lower(), parts[1].lower())] = parts[2:]
    sizes = [int(v) for v in info[("layout", "sizes")]]
    order = [v.lower() for v in info.get(("layout", "order"), ["bits", "x", "y", "z"])]
    bits = sizes[order.index("bits")] if "bits" in order else 16
    dims = [(o, n) for o, n in zip(order, sizes) if o != "bits"]
    fmt = info.get(("representation", "format"), ["integer"])[0].lower()
    sign = info.get(("representation", "sign"), ["unsigned"])[0].lower()
    comp = info.get(("representation", "compression"), ["uncompressed"])[0].lower()
    bo = info.get(("representation", "byte_order"), ["1"])
    ids = [path[:-4] + e for e in (".ids", ".IDS") if os.path.exists(path[:-4] + e)]
    data = open(ids[0], "rb").read() if ids else raw[m.end():] if m else b""
    if comp.startswith("gzip") or comp.startswith("compress"):
        k = data.find(b"\x1f\x8b")                        # skip anything before the gzip stream
        if 0 < k < 64:
            data = data[k:]
        for wbits in (16 + zlib.MAX_WBITS, zlib.MAX_WBITS, -zlib.MAX_WBITS):
            try:
                data = zlib.decompressobj(wbits).decompress(data)
                break
            except zlib.error:
                continue
        else:
            data = gzip.decompress(data)
    kind = "f" if fmt.startswith("real") or fmt.startswith("float") else ("i" if sign == "signed" else "u")
    dt = np.dtype(f"{'<' if bo[0] == '1' else '>'}{kind}{bits // 8}")
    n = int(np.prod([d for _, d in dims]))
    if len(data) < n * dt.itemsize:
        raise RuntimeError(f"{path}: expected {n * dt.itemsize} bytes of image data, found {len(data)}")
    arr = np.frombuffer(data[:n * dt.itemsize], dtype=dt).reshape([d for _, d in dims][::-1])
    return np.asarray(arr).squeeze()


def _read_any(f):
    lf = f.lower()
    if lf.endswith(".npy"):
        return np.load(f)
    if lf.endswith(".ics"):
        return read_ics(f)
    try:
        import tifffile
    except ImportError:
        pip_install("tifffile")
        import tifffile
    return tifffile.imread(f)


def load_volumes(folder, n, D, device, crop=0.5):
    """Load 3D volumes, e.g. BBBC024 / CytoPacq ground-truth masks. Accepts .tif/.tiff/.npy/.ics in any
    letter case, one file per volume or one 2D file per z-slice (slices are grouped by folder).
    Label masks (one integer per nucleus) become a 0/1 foreground mask; the central `crop` fraction in
    Y and X is kept (so a few nuclei fill the field, as in InverseBench); resized to D x 128 x 128, 0..1."""
    exts = (".tif", ".tiff", ".npy", ".ics")
    files = sorted(f for f in glob.glob(os.path.join(folder, "**", "*"), recursive=True)
                   if os.path.isfile(f) and f.lower().endswith(exts))
    log(f"  found {len(files)} image files in {folder}, e.g. {[os.path.relpath(f, folder) for f in files[:3]]}")
    if not files:
        raise RuntimeError(f"No .tif/.tiff/.npy/.ics files in {folder}")
    first = np.asarray(_read_any(files[0])).squeeze()
    log(f"  first file shape {first.shape}, dtype {first.dtype}")
    if first.ndim == 2:                                   # one 2D file per slice -> group by folder
        import re
        groups = {}
        for f in files:   # same folder + same name apart from the last number (the slice index) = one volume
            key = (os.path.dirname(f), re.sub(r"\d+(?=\D*$)", "", os.path.basename(f)))
            groups.setdefault(key, []).append(f)
        units = [sorted(g) for _, g in sorted(groups.items()) if len(g) > 1][:n]
        log(f"  2D slices -> {len(units)} volumes")
    else:
        units = [[f] for f in files[:n]]
    vols = []
    for unit in units:
        v = (np.stack([np.asarray(_read_any(f)).squeeze() for f in unit]) if len(unit) > 1
             else np.asarray(_read_any(unit[0])).squeeze())
        if v.ndim == 4:                                   # e.g. (z, y, x, channel)
            v = v[..., 0]
        assert v.ndim == 3, f"{unit[0]}: expected a 3D volume, got shape {v.shape}"
        if np.issubdtype(v.dtype, np.integer) and len(np.unique(v)) <= 256:
            v = v > 0                                     # label mask -> foreground
        v = torch.from_numpy(v.astype(np.float32))
        if crop < 1:
            _, Y, X = v.shape
            cy, cx = max(1, int(Y * crop)), max(1, int(X * crop))
            v = v[:, (Y - cy) // 2:(Y - cy) // 2 + cy, (X - cx) // 2:(X - cx) // 2 + cx]
        v = F.interpolate(v[None, None], size=(D, 128, 128), mode="trilinear", align_corners=False)[0, 0]
        v = (v - v.min()) / (v.max() - v.min()).clamp(min=1e-8)
        vols.append(v.to(device))
        log(f"  loaded {os.path.relpath(unit[0], folder)} -> {tuple(v.shape)}, "
            f"foreground {float((v > 0.5).float().mean()):.1%}")
    return vols

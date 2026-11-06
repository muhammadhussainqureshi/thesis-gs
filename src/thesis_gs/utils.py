"""Setup helpers: logging, package install, InverseBench clone, CaltechDATA download, LMDB reading, LR schedule."""
import glob
import math
import os
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np


CALTECH = "https://data.caltech.edu/records/zg89b-mpv16/files"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}


def pip_install(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)


try:
    import lmdb
except ImportError:
    pip_install("lmdb")
    import lmdb


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def ensure_inversebench(ib_dir):
    if not os.path.isdir(os.path.join(ib_dir, "inverse_problems")):
        log("Cloning InverseBench ->", ib_dir)
        subprocess.run(["git", "clone", "-q", "--depth", "1",
                        "https://github.com/devzhk/InverseBench.git", ib_dir], check=True)
    if ib_dir not in sys.path:
        sys.path.insert(0, ib_dir)


def ensure_lmdb_dataset(name, data_dir, local_dir):
    """Return a folder containing data.mdb for <name> (e.g. 'inv-scatter-test').
    Uses <data_dir>/<name>.zip if present, otherwise downloads it from CaltechDATA."""
    target = os.path.join(local_dir, name)
    if glob.glob(f"{target}/**/data.mdb", recursive=True):
        return os.path.dirname(glob.glob(f"{target}/**/data.mdb", recursive=True)[0])
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, f"{name}.zip")
    if not os.path.exists(zip_path):
        import requests
        log(f"Downloading {name}.zip from CaltechDATA ...")
        r = requests.get(f"{CALTECH}/{name}.zip?download=1", headers=UA, timeout=300)
        if r.status_code != 200 or r.content[:2] != b"PK":
            raise RuntimeError(f"Download of {name}.zip failed (HTTP {r.status_code}). Download it manually from "
                               f"https://data.caltech.edu/records/zg89b-mpv16 and put it at {zip_path}")
        with open(zip_path, "wb") as f:
            f.write(r.content)
    shutil.rmtree(target, ignore_errors=True)
    zipfile.ZipFile(zip_path).extractall(target)
    hits = glob.glob(f"{target}/**/data.mdb", recursive=True)
    if not hits:
        raise RuntimeError(f"No data.mdb inside {zip_path}")
    return os.path.dirname(hits[0])


def read_lmdb_images(folder, n, res=128):
    env = lmdb.open(folder, readonly=True, lock=False, create=False)
    with env.begin() as txn:
        total = txn.stat()["entries"]
        out = [np.frombuffer(txn.get(str(i).encode()), dtype=np.float32).reshape(res, res).copy()
               for i in range(min(n, total))]
    env.close()
    return out


def cosine_lr(opt, base, it, total):
    for pg in opt.param_groups:
        pg["lr"] = pg.get("base_lr", base) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * it / total)))

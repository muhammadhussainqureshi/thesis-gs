import gzip

import numpy as np

from thesis_gs.volumes import load_volumes, read_ics


def _write_ics(path, arr, newline="\r\n"):
    z, y, x = arr.shape
    nl = newline
    head = (f"\t{nl}ics_version\t2.0{nl}filename\tvol{nl}layout\tparameters\t4{nl}layout\torder\tbits\tx\ty\tz{nl}"
            f"layout\tsizes\t16\t{x}\t{y}\t{z}{nl}representation\tformat\tinteger{nl}representation\tsign\tunsigned{nl}"
            f"representation\tcompression\tgzip{nl}representation\tbyte_order\t1\t2{nl}end{nl}").encode()
    path.write_bytes(head + gzip.compress(arr.astype("<u2").tobytes()))


def test_read_ics_v2_gzip_crlf(tmp_path):
    lab = np.random.default_rng(0).integers(0, 21, (6, 10, 12)).astype(np.uint16)
    _write_ics(tmp_path / "a.ics", lab, "\r\n")
    assert np.array_equal(read_ics(str(tmp_path / "a.ics")), lab)


def test_load_volumes_label_mask(tmp_path):
    lab = np.zeros((8, 40, 40), np.uint16)
    lab[2:6, 10:30, 10:30] = 7
    _write_ics(tmp_path / "image-labels_0000.ics", lab, "\n")
    v = load_volumes(str(tmp_path), 1, 8, "cpu", crop=1.0)[0]
    assert v.shape == (8, 128, 128) and float(v.max()) == 1.0 and float(v.min()) == 0.0

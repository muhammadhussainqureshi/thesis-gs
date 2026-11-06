import os

import pytest
import torch

IB = os.environ.get("INVERSEBENCH_DIR", "/tmp/InverseBench")


@pytest.mark.skipif(not os.path.isdir(os.path.join(IB, "inverse_problems")), reason="InverseBench not cloned")
def test_slice_scatter_matches_inversebench():
    import sys
    sys.path.insert(0, IB)
    from thesis_gs.physics import SliceScatter
    op = SliceScatter(60, "cpu")
    f = torch.rand(128, 128)
    assert op.check(f) < 1e-5
    vol = torch.rand(3, 128, 128)
    assert op(vol).shape == (3, 20, 60)

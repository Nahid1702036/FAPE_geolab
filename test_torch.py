"""tests/test_torch.py — fape_geolab"""""
import pytest, numpy as np

torch_ok = False
try:
    import torch
    torch_ok = torch.cuda.is_available()
except ImportError:
    pass

def cube():
    v = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                  [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]], dtype=np.float32)
    f = np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7],[0,1,5],[0,5,4],
                  [2,3,7],[2,7,6],[0,3,7],[0,7,4],[1,2,6],[1,6,5]], dtype=np.int32)
    return v, f

@pytest.mark.skipif(not torch_ok, reason="CUDA unavailable")
class TestFapeTorch:
    def setup_method(self):
        from fape_geolab import FapeSurgicalPass
        self.layer = FapeSurgicalPass()
        v, f = cube()
        self.v = torch.tensor(v, device="cuda", requires_grad=True)
        self.f = torch.tensor(f, device="cuda")
        self.ac = torch.eye(4).unsqueeze(0).repeat(8,1,1).cuda()
        self.hc = torch.eye(4).unsqueeze(0).repeat(2,1,1).cuda()

    def test_shapes(self):
        ov, oval = self.layer(self.v, self.f, self.ac, self.hc)
        assert ov.shape == (8,3) and oval.shape == (8,)

    def test_validity_range(self):
        _, oval = self.layer(self.v, self.f, self.ac, self.hc)
        assert (oval >= 0).all() and (oval <= 1).all()

    def test_grad(self):
        ov, oval = self.layer(self.v, self.f, self.ac, self.hc)
        ((1-oval)*ov.norm(dim=-1)).mean().backward()
        assert self.v.grad is not None and not torch.isnan(self.v.grad).any()

    def test_deterministic(self):
        o1, _ = self.layer(self.v, self.f, self.ac, self.hc)
        o2, _ = self.layer(self.v, self.f, self.ac, self.hc)
        assert torch.allclose(o1, o2)

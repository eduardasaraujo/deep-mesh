import pytest
import torch

from deepmesh.networks import DeformationNet, MotionNet


def images(b=2, d=16, h=32, w=32):
    return {"sa": torch.rand(b, d, h, w), "2ch": torch.rand(b, 1, h, w), "4ch": torch.rand(b, 1, h, w)}


@pytest.mark.parametrize("views", [("sa",), ("sa", "2ch"), ("sa", "2ch", "4ch")])
def test_output_shapes_and_range(views):
    x = images()
    phi_d = DeformationNet(16, width=4, views=views)(x)
    phi_m = MotionNet(16, width=4, width_3d=2, views=views)(x, x)
    for phi in (phi_d, phi_m):
        assert phi.shape == (2, 3, 16, 32, 32)
        assert phi.abs().max() <= 1  # tanh: normalised displacement
        phi.sum().backward()  # gradients reach the zero-initialised head


def test_paper_sizes_match_reference_parameter_count():
    # counts of deformnet + Mesh_2d and MotionMesh_25d + Mesh_2d in the authors' code
    n = lambda m: sum(p.numel() for p in m.parameters())
    assert n(DeformationNet(64, 64)) == 38335043
    assert n(MotionNet(64, 64, 32)) == 21764323


def test_sax_view_is_required():
    with pytest.raises(ValueError):
        DeformationNet(16, 4, views=("2ch", "4ch"))


def test_initial_field_is_zero():
    x = images()
    assert DeformationNet(16, 4)(x).abs().max() == 0
    assert MotionNet(16, 4, 2)(x, x).abs().max() == 0


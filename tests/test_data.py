import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt

from deepmesh import mesh as M
from deepmesh.config import load_config
from deepmesh.data import MotionDataset, ReconstructionDataset, contour
from deepmesh.synthetic.phantom import generate_subject


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    import scripts.generate_synthetic as gen
    root = tmp_path_factory.mktemp("syn")
    cfg = load_config("configs/small.yaml", [f"data.root={root}", "data.n_frames=6", "data.mesh_around=32",
                                             "data.mesh_long=12"]).data
    rng = np.random.default_rng(0)
    topo = M.build_topology(cfg.mesh_around, cfg.mesh_long)
    gt_topo = M.build_topology(40, 15)
    tpl = M.lv_vertices(M.TEMPLATE_SHAPE, topo)
    np.savez(root / "template.npz", verts=tpl, faces=topo.faces)
    out = generate_subject(rng, cfg, topo, gt_topo, tpl)
    gen.write_subject(str(root / "train" / "s0"), out, gt_topo.faces)
    return root, cfg, out


def test_gt_mesh_lies_on_the_label_boundary(dataset):
    _, cfg, out = dataset
    A_inv = np.linalg.inv(out["affines"]["sa"])
    for t in (0, out["subject"]["es_frame"]):
        vox = out["gt_vertices"][t] @ A_inv[:3, :3].T + A_inv[:3, 3]
        for z in cfg.sa_slices[2:7]:
            pts = vox[np.abs(vox[:, 2] - z) < 1.0]  # coarse test mesh: wider band
            dist = distance_transform_edt(contour(out["labels"]["sa"][..., z, t]) == 0)
            d = dist[np.round(pts[:, 0]).astype(int), np.round(pts[:, 1]).astype(int)]
            assert len(pts) > 0 and np.median(d) <= 1.0


def test_dataset_items(dataset):
    root, cfg, _ = dataset
    item = ReconstructionDataset(str(root / "train"))[0]
    assert item["images"]["sa"].shape == (cfg.depth, cfg.height, cfg.width)
    assert item["images"]["2ch"].shape == (1, cfg.height, cfg.width)
    assert item["contour_sa"].shape == (cfg.height, cfg.width, cfg.depth)
    assert item["contour_sa"].sum() > 0 and item["contour_4ch"].sum() > 0
    m = MotionDataset(str(root / "train"), cfg.sa_slices, frame="es")[0]
    assert m["contour_sa"].shape == (cfg.height, cfg.width, len(cfg.sa_slices))
    assert m["ed_verts"].shape == m["gt_verts_t"].shape
    assert "gt_surface" in m


def test_es_biased_frame_sampling(dataset):
    root, cfg, out = dataset
    es = out["subject"]["es_frame"]
    ds = MotionDataset(str(root / "train"), cfg.sa_slices, use_gt_ed_mesh=True, es_prob=1.0, es_window=1)
    frames = {ds[0]["frame"] for _ in range(20)}
    assert frames <= {(es - 1) % cfg.n_frames, es, (es + 1) % cfg.n_frames}

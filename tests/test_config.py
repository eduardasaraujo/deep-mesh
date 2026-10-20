from deepmesh.config import load_config


def test_config_overrides():
    cfg = load_config("configs/small.yaml", ["recon.lr=1e-3", "model.views=[sa]", "data.root=/x/y"])
    assert cfg.recon.lr == 1e-3 and cfg.model.views == ["sa"] and cfg.data.root == "/x/y"

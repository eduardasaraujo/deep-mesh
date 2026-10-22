"""Convert segmented native cines (scripts/segment_real.py) into a DeepMesh dataset.

    python scripts/prepare_real.py <native_root> --config configs/real.yaml

Each subject goes to train/val/test according to a hash of its name (the same split
on every run), and subjects already prepared or rejected are skipped, so the script
can be stopped and resumed. Subjects that fail quality control are listed in
<data.root>/qc.csv and not written.
"""
import csv
import json
import os
import sys
import zlib
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deepmesh import mesh as M  # noqa: E402
from deepmesh.config import base_parser, load_config  # noqa: E402
from deepmesh.real.prepare import prepare_subject, write_subject  # noqa: E402

_STATE = {}


def split_of(name, fractions):
    """Deterministic split from a hash of the subject name."""
    u = (zlib.crc32(name.encode()) % 10000) / 10000
    if u < fractions[0]:
        return "train"
    return "val" if u < fractions[0] + fractions[1] else "test"


def _init(d, template, fractions):
    _STATE.update(d=d, template=template, fractions=fractions)


def _work(native_dir):
    d, name = _STATE["d"], os.path.basename(native_dir)
    qc_path = os.path.join(d.root, "_qc", f"{name}.json")
    if os.path.exists(qc_path):  # done in an earlier run
        with open(qc_path) as f:
            return json.load(f)
    try:
        out, qc = prepare_subject(native_dir, d, _STATE["template"])
    except Exception as e:  # corrupt or unusual series: report and continue
        out, qc = None, dict(reject=f"error: {e}")
    qc = dict(name=name, split=split_of(name, _STATE["fractions"]), **qc)
    if out is not None and not qc["reject"]:
        out["meta"]["qc"] = qc
        write_subject(os.path.join(d.root, qc["split"], name), out)
    with open(qc_path, "w") as f:  # written last: marks the subject as done
        json.dump(qc, f)
    return qc


def main():
    p = base_parser(__doc__)
    p.add_argument("native_root")
    p.add_argument("--splits", nargs=3, type=float, default=[0.8, 0.1, 0.1], metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--workers", type=int, default=os.cpu_count())
    args = p.parse_args()
    cfg = load_config(args.config, args.set)
    d = cfg.data
    fractions = [f / sum(args.splits) for f in args.splits]

    subjects = sorted(os.path.join(args.native_root, n) for n in os.listdir(args.native_root)
                      if os.path.exists(os.path.join(args.native_root, n, "seg_sa.nii.gz")))
    for folder in ("_qc", "train", "val", "test"):
        os.makedirs(os.path.join(d.root, folder), exist_ok=True)
    template = M.save_template(os.path.join(d.root, "template.npz"), d.mesh_around, d.mesh_long)
    rows = []
    with Pool(args.workers, _init, (d, template, fractions)) as pool:
        for i, row in enumerate(pool.imap_unordered(_work, subjects), 1):
            rows.append(row)
            print(f"[{i}/{len(subjects)}] {row['name']}: {row['reject'] or row['split']}", flush=True)

    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "name", k != "split", k != "reject", k))
    with open(os.path.join(d.root, "qc.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["name"]))
    passed = [r for r in rows if not r["reject"]]
    counts = {s: sum(r["split"] == s for r in passed) for s in ("train", "val", "test")}
    print(f"{len(passed)}/{len(rows)} subjects passed QC -> "
          + ", ".join(f"{k}: {v}" for k, v in counts.items()) + f"; report in {d.root}/qc.csv")


if __name__ == "__main__":
    main()

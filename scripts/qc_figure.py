"""Quality-control figure for one prepared subject (before training).

Rows: ED with the GT surface, ES with the GT surface, ED with the aligned template.
Columns: basal / mid / apical SAX slices, 2CH, 4CH. Green: segmentation contour.

    python scripts/qc_figure.py <subject_dir> <out.png> --config configs/real.yaml
"""
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deepmesh.config import base_parser, load_config  # noqa: E402
from deepmesh.data import Subject, contour  # noqa: E402
from deepmesh.metrics import mesh_contour_points  # noqa: E402


def main():
    p = base_parser(__doc__)
    p.add_argument("subject")
    p.add_argument("out")
    args = p.parse_args()
    cfg = load_config(args.config, args.set)
    s = Subject(args.subject)
    affines = {k: v.numpy().astype(np.float64) for k, v in s.affines().items()}
    es, sl = s.meta["es_frame"], cfg.data.sa_slices
    planes = [("SAX basal", "sa", sl[1]), ("SAX médio", "sa", sl[len(sl) // 2]),
              ("SAX apical", "sa", sl[-2]), ("2CH", "2ch", 0), ("4CH", "4ch", 0)]
    rows = [(0, s.array("gt_surface_ed.npz"), "ED, superfície verdadeira", "red"),
            (es, s.array("gt_surface_es.npz"), f"ES (t={es}), superfície verdadeira", "red"),
            (0, s.array("template_aligned.npy"), "ED, template alinhado", "cyan")]
    fig, axes = plt.subplots(len(rows), len(planes), figsize=(3.2 * len(planes), 3.3 * len(rows)))
    for r, (t, verts, title, color) in enumerate(rows):
        for c, (name, view, z) in enumerate(planes):
            img, lab = s.frame(view, t), s.label(view, t)
            if view == "sa":
                img, lab = img[..., z], lab[..., z]
            ax = axes[r, c]
            ax.imshow(img, cmap="gray")
            gt = np.argwhere(contour(lab))
            ax.plot(gt[:, 1], gt[:, 0], "s", color="lime", ms=1)
            pts = mesh_contour_points(verts.astype(np.float64), affines[f"{view}_inv"], z, 0.5, *img.shape)
            ax.plot(pts[:, 1], pts[:, 0], ".", color=color, ms=1.5)
            ax.set_title(f"{name} | {title}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f"{s.name}  qc: {s.meta.get('qc', {})}", fontsize=7)
    fig.tight_layout()
    fig.savefig(args.out, dpi=80)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

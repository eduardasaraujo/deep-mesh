"""Figures in the style of the paper for one test subject.

    contours.png    predicted (red) vs GT (green) contours on SAX/2CH/4CH over the cycle (Fig. 7)
    cycle.gif       the same overlay as an animation
    meshes.png      predicted vs GT meshes at ED and ES (Fig. 6a / Fig. 9)
    soft_maps.png   probability maps produced by the differentiable rasterizer (Fig. 5)
    curves.png      training curves of both modules
    obj/            predicted mesh sequence as OBJ files (optional, --obj)

Run scripts/test_motion.py first: it writes the predicted sequences.
"""
import csv
import os
import sys

import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deepmesh.config import base_parser, load_config  # noqa: E402
from deepmesh.data import Subject, contour, load_template  # noqa: E402
from deepmesh.metrics import mesh_contour_points  # noqa: E402
from deepmesh.mesh import save_obj  # noqa: E402
from deepmesh.rasterizer import plane_probability, soft_contour_map, to_plane  # noqa: E402


def view_planes(cfg):
    sa = cfg.data.sa_slices
    return [("SAX basal", "sa", sa[1]), ("SAX médio", "sa", sa[4]), ("SAX apical", "sa", sa[7]),
            ("2CH", "2ch", 0), ("4CH", "4ch", 0)]


def overlay(ax, subj, affines, verts, view, z, t, band):
    img, lab = subj.frame(view, t), subj.label(view, t)
    if view == "sa":
        img, lab = img[..., z], lab[..., z]
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    gt = np.argwhere(contour(lab))
    ax.plot(gt[:, 1], gt[:, 0], "s", color="lime", ms=1.2, alpha=0.8)
    pts = mesh_contour_points(verts, affines[f"{view}_inv"], z, band, *img.shape)
    ax.plot(pts[:, 1], pts[:, 0], "o", color="red", ms=1.2)
    ax.set_xticks([])
    ax.set_yticks([])


def contour_figure(subj, affines, seq, cfg, path):
    T, es = len(seq), subj.meta["es_frame"]
    frames = sorted({0, T // 8, es // 2, es, (es + T) // 2, T - 1})
    planes = view_planes(cfg)
    fig, axes = plt.subplots(len(planes), len(frames), figsize=(2 * len(frames), 2 * len(planes)))
    for r, (title, view, z) in enumerate(planes):
        for c, t in enumerate(frames):
            overlay(axes[r, c], subj, affines, seq[t], view, z, t, cfg.eval.slice_band)
            if r == 0:
                axes[r, c].set_title(f"t={t}" + (" (ES)" if t == es else " (ED)" if t == 0 else ""))
        axes[r, 0].set_ylabel(title)
    fig.suptitle("vermelho: malha prevista   verde: contorno verdadeiro")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def cycle_gif(subj, affines, seq, cfg, path):
    planes = [p for p in view_planes(cfg) if p[0] in ("SAX médio", "2CH", "4CH")]
    frames = []
    for t in range(len(seq)):
        fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))
        for ax, (title, view, z) in zip(axes, planes):
            overlay(ax, subj, affines, seq[t], view, z, t, cfg.eval.slice_band)
            ax.set_title(f"{title}  t={t}", fontsize=9)
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3])
        plt.close(fig)
    imageio.mimsave(path, frames, duration=0.12, loop=0)


def mesh_figure(subj, seq, faces, path):
    es = subj.meta["es_frame"]
    gt_ed = np.load(os.path.join(subj.path, "gt_surface_ed.npz"))
    gt_es = np.load(os.path.join(subj.path, "gt_surface_es.npz"))
    centre = seq[0].mean(0)
    axis = np.linalg.svd(seq[0] - centre)[2][0]  # long axis ~ first principal direction
    side = np.cross(axis, [0, 0, 1.0])
    side /= np.linalg.norm(side)
    basis = np.stack([side, np.cross(axis, side), axis], 1)
    items = [("prevista ED", seq[0], faces, "tab:red"), ("verdadeira ED", gt_ed["verts"], gt_ed["faces"], "tab:green"),
             (f"prevista ES (t={es})", seq[es], faces, "tab:red"),
             (f"verdadeira ES (t={es})", gt_es["verts"], gt_es["faces"], "tab:green")]
    fig = plt.figure(figsize=(13, 4))
    for i, (title, v, f, color) in enumerate(items):
        ax = fig.add_subplot(1, 4, i + 1, projection="3d")
        p = (v - centre) @ basis
        ax.plot_trisurf(p[:, 0], p[:, 1], p[:, 2], triangles=f, color=color, alpha=0.6, linewidth=0)
        ax.set_title(title)
        ax.set_box_aspect((1, 1, 1.4))
        lim = 55
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim))
        ax.view_init(elev=10, azim=-60)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def soft_map_figure(subj, affines, verts, cfg, path):
    H, W = cfg.data.height, cfg.data.width
    planes = [p for p in view_planes(cfg) if p[0] in ("SAX médio", "2CH", "4CH")]
    v = torch.as_tensor(verts, dtype=torch.float32)[None]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    for ax, (title, view, z) in zip(axes, planes):
        pts, k = to_plane(v, torch.as_tensor(affines[f"{view}_inv"], dtype=torch.float32)[None], H, W)
        prob = plane_probability(k, z, cfg.recon.tau)
        ax.imshow(soft_contour_map(pts[0], prob[0], H, W, cfg.recon.tau).numpy(), cmap="magma", vmin=0, vmax=1)
        ax.set_title(f"{title}: p = exp(-τd²)")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def curves_figure(cfg, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, module in zip(axes, ("reconstruction", "motion")):
        p = os.path.join(cfg.output.runs, module, "history.csv")
        if not os.path.exists(p):
            continue
        rows = list(csv.DictReader(open(p)))
        epochs = [int(r["epoch"]) for r in rows]
        for key in [k for k in rows[0] if k.startswith("train_") or k.startswith("val_")]:
            if key.endswith("total") or key.endswith("shape"):
                ax.plot(epochs, [float(r[key]) for r in rows], label=key, ls="-" if "train" in key else "--")
        ax.set_title(module)
        ax.set_xlabel("época")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    p = base_parser(__doc__)
    p.add_argument("--subject", default=None, help="test subject name (default: first)")
    p.add_argument("--obj", action="store_true", help="also export the sequence as OBJ files")
    args = p.parse_args()
    cfg = load_config(args.config, args.set)

    seq_dir = os.path.join(cfg.output.results, "motion", "sequences")
    name = args.subject or sorted(os.listdir(seq_dir))[0][:-4]
    subj = Subject(os.path.join(cfg.data.root, "test", name))
    seq = np.load(os.path.join(seq_dir, f"{name}.npy")).astype(np.float64)
    affines = {k: v.numpy().astype(np.float64) for k, v in subj.affines().items()}
    faces = load_template(cfg.data.root)[0].numpy()

    out = os.path.join(cfg.output.results, "figures", name)
    os.makedirs(out, exist_ok=True)
    contour_figure(subj, affines, seq, cfg, os.path.join(out, "contours.png"))
    cycle_gif(subj, affines, seq, cfg, os.path.join(out, "cycle.gif"))
    mesh_figure(subj, seq, faces, os.path.join(out, "meshes.png"))
    soft_map_figure(subj, affines, seq[0], cfg, os.path.join(out, "soft_maps.png"))
    curves_figure(cfg, os.path.join(out, "curves.png"))
    if args.obj:
        os.makedirs(os.path.join(out, "obj"), exist_ok=True)
        for t, v in enumerate(seq):
            save_obj(os.path.join(out, "obj", f"frame_{t:02d}.obj"), v, faces)
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()

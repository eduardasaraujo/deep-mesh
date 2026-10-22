"""Forward passes and objectives of the two DeepMesh modules (Fig. 1)."""
import torch

from deepmesh.geometry import displace_vertices
from deepmesh.losses import chamfer_distance, huber_regularization, laplacian_smoothing, shape_loss, similarity_loss
from deepmesh.networks import DeformationNet, MotionNet


def to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    return x


def shape_views(cfg):
    """Views supervised by L_shape; defaults to the input views (model.shape_views: null)."""
    return getattr(cfg.model, "shape_views", None) or cfg.model.views


def sa_shape(cfg):
    return (cfg.data.height, cfg.data.width, cfg.data.depth)


def build_deformation_net(cfg):
    return DeformationNet(cfg.data.depth, cfg.model.width, cfg.model.views)


def build_motion_net(cfg):
    return MotionNet(cfg.data.depth, cfg.model.width, cfg.model.width_3d, cfg.model.views)


# ----------------------------------------------------------------- reconstruction
def reconstruct(net, batch, cfg):
    """Eq. (1)-(2): V0 = V_tpl + S(H_D(I_0), V_tpl)."""
    field = net(batch["images"])
    a = batch["affines"]
    verts = displace_vertices(batch["template"], field, a["sa"], a["sa_inv"], sa_shape(cfg))
    return verts, field


def reconstruction_loss(verts, field, batch, edges, cfg):
    """Eq. (11): L_recon = L_shape + l1 L_smooth + b1 L_surf + g1 L_reg."""
    c = cfg.recon
    if getattr(c, "use_acquired_slices", False):  # real data: interpolated slices are not labels
        slices = list(cfg.data.sa_slices)
    else:
        slices = list(range(0, cfg.data.depth, c.sa_slice_stride))
    contours = {"sa": batch["contour_sa"][..., slices], "2ch": batch["contour_2ch"], "4ch": batch["contour_4ch"]}
    l_shape, _ = shape_loss(verts, batch["affines"], contours, slices, c.tau, shape_views(cfg))
    terms = dict(shape=l_shape, smooth=laplacian_smoothing(verts, edges),
                 surf=chamfer_distance(verts, batch["gt_surface"]), reg=huber_regularization(field))
    total = (c.shape_weight * terms["shape"] + c.lambda1 * terms["smooth"]
             + c.beta1 * terms["surf"] + c.gamma1 * terms["reg"])
    return total, terms


# ----------------------------------------------------------------- motion
def track(net, batch, cfg, ed_verts=None):
    """Eq. (6)-(7): V_t = V0 + S(H_M(I_0, I_t), V0)."""
    field = net(batch["images_ed"], batch["images_t"])
    a = batch["affines"]
    ed_verts = batch["ed_verts"] if ed_verts is None else ed_verts
    verts = displace_vertices(ed_verts, field, a["sa"], a["sa_inv"], sa_shape(cfg))
    return verts, field


def motion_loss(verts, field, batch, edges, cfg):
    """Eq. (12): L_motion = L_shape + l2 L_smooth + b2 L_sim + g2 L_reg."""
    c = cfg.motion
    contours = {"sa": batch["contour_sa"], "2ch": batch["contour_2ch"], "4ch": batch["contour_4ch"]}
    l_shape, _ = shape_loss(verts, batch["affines"], contours, cfg.data.sa_slices, c.tau, shape_views(cfg))
    terms = dict(shape=l_shape, smooth=laplacian_smoothing(verts, edges),
                 sim=similarity_loss(batch["images_ed"]["sa"], batch["images_t"]["sa"], field),
                 reg=huber_regularization(field))
    total = (c.shape_weight * terms["shape"] + c.lambda2 * terms["smooth"]
             + c.beta2 * terms["sim"] + c.gamma2 * terms["reg"])
    return total, terms

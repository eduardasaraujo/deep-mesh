"""Myocardium segmentation with the pre-trained networks of Bai et al. (2018).

The models come from https://github.com/baiwenjia/ukbb_cardiac (Apache-2.0) and
were trained on UK Biobank. They are TensorFlow 1 checkpoints, loaded here with
tf.compat.v1; inference follows common/deploy_network.py of that repository
(1-99 percentile rescaling, padding to a multiple of 16, slice-wise prediction).

Label maps:
    FCN_sa           0 bg, 1 LV blood pool, 2 LV myocardium, 3 RV
    FCN_la_4ch_seg4  0 bg, 1 LV, 2 myocardium, 3 RV, 4 LA, 5 RA
    FCN_la_2ch       0 bg, 1 LA  (no ventricle: the 2CH view is not segmented here)

TensorFlow is imported lazily so that the rest of the package works without it.
"""
import math
import os
import urllib.request

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

MODEL_URL = "https://www.doc.ic.ac.uk/~wbai/data/ukbb_cardiac/trained_model/"
MODELS = {"sa": "FCN_sa", "la_4ch": "FCN_la_4ch_seg4"}
MYO_LABEL = {"sa": 2, "la_4ch": 2}
LV_LABEL = {"sa": 1, "la_4ch": 1}
# in-plane pixel size of the UK Biobank cines the networks were trained on (mm);
# other datasets (e.g. Kaggle DSB, 0.6-1.9 mm) are resampled to it before inference
TRAIN_SPACING = 1.8


def download_models(model_dir, names=MODELS.values()):
    os.makedirs(model_dir, exist_ok=True)
    for name in names:
        for ext in ("meta", "index", "data-00000-of-00001"):
            path = os.path.join(model_dir, f"{name}.{ext}")
            if not os.path.exists(path):
                urllib.request.urlretrieve(f"{MODEL_URL}{name}.{ext}", path)


def rescale_intensity(image, thres=(1.0, 99.0)):
    lo, hi = np.percentile(image, thres)
    return (np.clip(image, lo, hi).astype(np.float32) - lo) / max(hi - lo, 1e-6)


class Segmenter:
    """Holds one TF1 graph; call it on a (X, Y, Z, T) image to get a label map of the same shape."""

    def __init__(self, model_path):
        import tensorflow as tf
        tf1 = tf.compat.v1
        tf1.disable_eager_execution()
        self.graph = tf1.Graph()
        meta = tf1.MetaGraphDef()
        with open(f"{model_path}.meta", "rb") as f:
            meta.ParseFromString(f.read())
        for node in meta.graph_def.node:
            # shapes cached by TF 1.x are rejected by recent TF versions; they are recomputed
            node.attr.pop("_output_shapes", None)
        with self.graph.as_default():
            self.sess = tf1.Session(graph=self.graph)
            saver = tf1.train.import_meta_graph(meta)
            saver.restore(self.sess, model_path)

    def __call__(self, image, spacing=None):
        """image (X, Y, Z, T); spacing: in-plane pixel size (mm), resampled to TRAIN_SPACING if given."""
        if spacing is not None:
            factors = (spacing[0] / TRAIN_SPACING, spacing[1] / TRAIN_SPACING, 1, 1)
            if max(abs(f - 1) for f in factors) > 0.05:
                small = zoom(image.astype(np.float32), factors, order=1)
                pred = self(small)
                back = [image.shape[i] / small.shape[i] for i in range(2)] + [1, 1]
                return zoom(pred, back, order=0)
        X, Y, Z, T = image.shape
        image = rescale_intensity(image)
        X2, Y2 = int(math.ceil(X / 16.0)) * 16, int(math.ceil(Y / 16.0)) * 16
        x0, y0 = (X2 - X) // 2, (Y2 - Y) // 2
        image = np.pad(image, ((x0, X2 - X - x0), (y0, Y2 - Y - y0), (0, 0), (0, 0)))
        pred = np.zeros((X, Y, Z, T), np.uint8)
        for t in range(T):
            batch = np.transpose(image[..., t], (2, 0, 1))[..., None].astype(np.float32)
            out = self.sess.run("pred:0", feed_dict={"image:0": batch, "training:0": False})
            pred[..., t] = np.transpose(out, (1, 2, 0))[x0:x0 + X, y0:y0 + Y]
        return pred

    def close(self):
        self.sess.close()


def segment_subjects(native_root, model_dir, views=("sa", "la_4ch"), overwrite=False):
    """Write seg_<view>.nii.gz next to every <view>.nii.gz under native_root/<subject>/."""
    subjects = sorted(d for d in os.listdir(native_root) if os.path.isdir(os.path.join(native_root, d)))
    for view in views:
        seg = Segmenter(os.path.join(model_dir, MODELS[view]))
        for name in subjects:
            src = os.path.join(native_root, name, f"{view}.nii.gz")
            dst = os.path.join(native_root, name, f"seg_{view}.nii.gz")
            if not os.path.exists(src) or (os.path.exists(dst) and not overwrite):
                continue
            nim = nib.load(src)
            labels = seg(np.asanyarray(nim.dataobj).astype(np.float32), nim.header.get_zooms()[:2])
            nib.save(nib.Nifti1Image(labels, nim.affine), dst)
            print(f"{name} {view}: labels {np.unique(labels).tolist()}")
        seg.close()

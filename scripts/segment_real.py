"""Segment native NIfTI cines (sa.nii.gz, la_4ch.nii.gz) with the networks of Bai et al.

Needs TensorFlow (preinstalled on Colab). Writes seg_sa.nii.gz and seg_la_4ch.nii.gz
next to the images; already segmented subjects are skipped.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deepmesh.real.segment import download_models, segment_subjects  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("native_root", help="folder with one subfolder per subject")
    p.add_argument("--models", default="models/ukbb_cardiac", help="where the pre-trained models are stored")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    download_models(args.models)
    segment_subjects(args.native_root, args.models, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

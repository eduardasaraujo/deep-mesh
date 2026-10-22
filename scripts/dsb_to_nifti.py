"""Convert the Kaggle Data Science Bowl Cardiac studies (DICOM) into native NIfTI cines.

    python scripts/dsb_to_nifti.py <zip_or_folder> [...] --out <native_root>

Accepts the zip downloaded from Kaggle (read study by study, nested zips included)
or an already extracted folder. Writes <native_root>/dsb_<id>/{sa,la_2ch,la_4ch}.nii.gz
and <native_root>/convert.csv. Converted studies are skipped, so it can be resumed.
"""
import argparse
import csv
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deepmesh.real.dicom import convert_study, convert_zip, find_studies, is_converted  # noqa: E402


def _work(args):
    name, study, out_dir = args
    if is_converted(out_dir):
        return name, "ok (done before)"
    try:
        return name, convert_study(study, out_dir)
    except Exception as e:  # irregular study: report and continue
        return name, f"error: {e}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+", help="Kaggle zip file(s) or extracted folder(s)")
    p.add_argument("--out", required=True, help="native NIfTI root")
    p.add_argument("--work", default="/tmp/dsb_work", help="scratch folder for one study at a time")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for src in args.inputs:
        if src.lower().endswith(".zip"):
            rows += convert_zip(src, args.out, args.work)
            continue
        jobs = [(n, s, os.path.join(args.out, n)) for n, s in sorted(find_studies(src).items())]
        with Pool(args.workers) as pool:
            for i, (name, status) in enumerate(pool.imap_unordered(_work, jobs), 1):
                rows.append((name, status))
                print(f"[{i}/{len(jobs)}] {name}: {status}", flush=True)
    with open(os.path.join(args.out, "convert.csv"), "w", newline="") as f:
        csv.writer(f).writerows([("name", "status")] + sorted(rows))
    ok = sum(s.startswith("ok") for _, s in rows)
    print(f"{ok}/{len(rows)} studies converted")


if __name__ == "__main__":
    main()

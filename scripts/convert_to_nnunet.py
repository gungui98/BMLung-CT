"""Convert kc-cancer-final → nnUNetv2 raw format.

Output layout (under ``--out-root``):
    Dataset701_KCCancer/
        imagesTr/<case>_0000.nii.gz   (copy of train images)
        labelsTr/<case>.nii.gz        (union of all per-lesion masks)
        imagesTs/<case>_0000.nii.gz   (copy of test images)
        labelsTs/<case>.nii.gz        (union — optional, used for eval)
        dataset.json

Only train+test cases that have at least one lesion annotation are exported
(``healthy`` cases without any mask get an all-zero label, which is fine for
nnUNet but burns disk; we skip them by default).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[1] / "src"))

from bmct.data.schema import load_case_table, load_lesion_table  # noqa: E402
from bmct.data.preprocessing import load_ct_volume               # noqa: E402


def export_case(case, lesions, src_root: Path, dst_images: Path, dst_labels: Path,
                include_negatives: bool) -> bool:
    img_src = src_root / case.series_path
    if not img_src.exists():
        return False
    nii = nib.load(str(img_src))
    # Go through load_ct_volume so the HU offset auto-fix is applied for any
    # case stored without a proper -1024 intercept (no-op on already-clean
    # data). Volume is returned as (Z, Y, X); transpose back to (X, Y, Z) so
    # the saved NIfTI matches the original orientation + affine.
    vol_zyx, _ = load_ct_volume(str(img_src))
    arr = np.transpose(vol_zyx, (2, 1, 0)).astype(np.int16)
    union = np.zeros_like(arr, dtype=np.uint8)
    any_mask = False
    for lez in lesions:
        mpath = src_root / lez.mask_path
        if not mpath.exists():
            continue
        m = nib.load(str(mpath)).get_fdata() > 0.5
        if m.shape != arr.shape:
            continue
        union = np.maximum(union, m.astype(np.uint8))
        any_mask = True
    if not any_mask and not include_negatives:
        return False

    out_img = dst_images / f"{case.case_id}_0000.nii.gz"
    out_lab = dst_labels / f"{case.case_id}.nii.gz"
    out_img.parent.mkdir(parents=True, exist_ok=True)
    out_lab.parent.mkdir(parents=True, exist_ok=True)

    # Save image (re-write to ensure clean dtype) + label NIfTI.
    nib.Nifti1Image(arr, nii.affine, nii.header).to_filename(out_img)
    nib.Nifti1Image(union, nii.affine).to_filename(out_lab)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--data-cfg", default="configs/data/kc_cancer.yaml")
    ap.add_argument("--out-root", required=True,
                    help="nnUNet_raw base dir; will create Dataset701_KCCancer/")
    ap.add_argument("--dataset-id", type=int, default=701)
    ap.add_argument("--dataset-name", default="KCCancer")
    ap.add_argument("--include-negatives", action="store_true",
                    help="export healthy / no-lesion cases too (all-zero label)")
    ap.add_argument("--split-col", default="split",
                    help="clinical.csv column for split labels (default: split). "
                         "Use split_v2 for the cohort-stratified resplit.")
    args = ap.parse_args()

    root = Path(args.dataset_root)
    from omegaconf import OmegaConf
    data_cfg = OmegaConf.load(args.data_cfg)
    cases = load_case_table(str(root / data_cfg.manifest))
    # Every lesion from every reader: the reference is the union of both test
    # readings. primary_only=True would keep only the first lesion of each test
    # case, because is_primary marks L1 alone in the test split.
    lesion_table = load_lesion_table(str(root / data_cfg.strong_manifest), primary_only=False)

    dst = Path(args.out_root) / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    dst.mkdir(parents=True, exist_ok=True)

    # Pick the split column. If split_v2, treat test_internal+test_ood as imagesTs
    # and keep them separable via subfolders.
    if args.split_col == "split_v2":
        # need raw csv access since CaseRecord only has `split`.
        clin = pd.read_csv(root / data_cfg.manifest)
        v2 = dict(zip(clin["case_id"], clin["split_v2"]))
        for c in cases:
            c.split = v2.get(c.case_id, c.split)         # mutate in-place
        train_cases = [c for c in cases if c.split in ("train", "val")]
        test_cases  = [c for c in cases if c.split in ("test_internal", "test_ood")]
    else:
        train_cases = [c for c in cases if c.split in ("train", "val")]
        test_cases  = [c for c in cases if c.split == "test"]

    n_train_kept = 0
    for c in tqdm(train_cases, desc="train"):
        lesions = lesion_table.get(c.case_id, [])
        kept = export_case(c, lesions, root,
                            dst / "imagesTr", dst / "labelsTr",
                            include_negatives=args.include_negatives)
        n_train_kept += int(kept)
    n_test_kept = 0
    for c in tqdm(test_cases, desc="test"):
        lesions = lesion_table.get(c.case_id, [])
        # When using split_v2 we put internal vs ood into separate folders so
        # nnUNetv2_predict can be run independently on each.
        if args.split_col == "split_v2" and c.split == "test_ood":
            img_dst = dst / "imagesTs_ood"
            lab_dst = dst / "labelsTs_ood"
        elif args.split_col == "split_v2" and c.split == "test_internal":
            img_dst = dst / "imagesTs_internal"
            lab_dst = dst / "labelsTs_internal"
        else:
            img_dst = dst / "imagesTs"
            lab_dst = dst / "labelsTs"
        kept = export_case(c, lesions, root, img_dst, lab_dst,
                            include_negatives=args.include_negatives)
        n_test_kept += int(kept)

    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": n_train_kept,
        "file_ending": ".nii.gz",
        "description": "kc-cancer-final lung lesion seg (single-class binary).",
    }
    (dst / "dataset.json").write_text(json.dumps(dataset_json, indent=2))
    print(f"[done] {dst}  train={n_train_kept}  test={n_test_kept}")


if __name__ == "__main__":
    main()

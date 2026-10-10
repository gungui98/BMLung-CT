# BMLung-CT baseline code

Code for the baseline benchmarks reported in *BMLung-CT: a Vietnamese routine chest CT
dataset with lesion annotations and baseline benchmarks for lung lesion segmentation
and malignancy assessment*.

This repository contains code and the predefined data split only. It contains no images,
masks, labels or patient information. The dataset is available from the corresponding
author on reasonable request, subject to a data-use agreement and approval by the Bach
Mai Hospital Ethics Board.

## Contents

| Path | What it is |
|---|---|
| `splits/split_manifest.csv` | The predefined split: one row per examination, `case_id` and `split` (`train` 1,800, `val` 200, `test` 252). |
| `src/bmct/` | Data loading and preprocessing, models, losses and metrics. |
| `configs/` | Data, model and training configuration for every baseline in the paper. |
| `scripts/` | Training, inference and evaluation entry points (below). |

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

The paper used Python 3.12, PyTorch 2.10, torchvision 0.25, timm 1.0,
segmentation-models-pytorch 0.5, nnU-Net 2.7 and scikit-learn 1.8.

## Data layout

Scripts take the dataset directory with `--dataset-root`. It is expected to contain

```
clinical.csv                                     one row per case: case_id, split, label, ...
lesions.csv                                      one row per (case, lesion, reader)
train/images/<case_id>.nii.gz                    CT, int16 HU
train/lesions/<case_id>/<lesion_id>/<reader>.nii.gz
test/images/<case_id>.nii.gz
test/lesions/<case_id>/<lesion_id>/<reader>.nii.gz
```

The `split` column must match `splits/split_manifest.csv`.

## Reproducing the benchmarks

**Stage 1, segmentation (2D U-Net, U-Net++, DeepLabV3+).**

```bash
python scripts/train_slice.py --dataset-root $DATA --config configs/experiment/kc_cancer_unet2d_slice.yaml
python scripts/train_slice.py --dataset-root $DATA --config configs/experiment/kc_cancer_unetpp.yaml
python scripts/train_slice.py --dataset-root $DATA --config configs/experiment/kc_cancer_deeplabv3p.yaml
python scripts/evaluate_seg_slice.py --dataset-root $DATA --config <experiment.yaml> --checkpoint <run>/best.pt --split test
```

**Stage 1, nnU-Net v2 2D.** Export to nnU-Net format, then use the standard nnU-Net
commands. The test labels are the union of every annotated lesion from both readers.

```bash
python scripts/convert_to_nnunet.py --dataset-root $DATA --data-cfg configs/data/kc_cancer.yaml --out-root $nnUNet_raw
nnUNetv2_plan_and_preprocess -d 701 --verify_dataset_integrity
nnUNetv2_train 701 2d 0 --npz -tr nnUNetTrainer_250epochs
nnUNetv2_predict -i $nnUNet_raw/Dataset701_KCCancer/imagesTs -o $PRED -d 701 -c 2d -f 0 \
    -tr nnUNetTrainer_250epochs -chk checkpoint_best.pth
python scripts/boundary_metrics.py --pred-dir $PRED --gt-dir $nnUNet_raw/Dataset701_KCCancer/labelsTs \
    --out-dir $OUT --tolerance-mm 1 2 3
```

**Stage 2, lesion classifiers (ResNet-18, ResNet-50, DenseNet-121, ConvNeXt-Tiny)** on
ground-truth lesion crops:

```bash
python scripts/train_nodule.py --dataset-root $DATA --config configs/experiment/kc_cancer_nodule_clf.yaml
python scripts/train_nodule.py --dataset-root $DATA --config configs/experiment/kc_cancer_nodule_clf_resnet50.yaml
python scripts/train_nodule.py --dataset-root $DATA --config configs/experiment/kc_cancer_nodule_clf_densenet121.yaml
python scripts/train_nodule.py --dataset-root $DATA --config configs/experiment/kc_cancer_nodule_clf_convnext.yaml
python scripts/evaluate_nodule.py --dataset-root $DATA --config <experiment.yaml> --checkpoint <run>/best.pt --split test
```

**End to end.** Candidates are the 3-D connected components (6-connectivity) of the
thresholded (0.5) segmentation map with at least 50 voxels; each is classified, and the
case-level probability is the maximum over candidates, or 0 if there is none.

```bash
python scripts/inference_e2e.py --dataset-root $DATA --data-cfg configs/data/kc_cancer.yaml \
    --seg-ckpt <seg run>/best.pt --seg-model-cfg configs/model/deeplabv3p_2d.yaml \
    --nodule-ckpt <clf run>/best.pt --nodule-model-cfg configs/model/nodule_classifier_resnet50.yaml \
    --split test --out-dir runs/e2e_combos/seg-deeplab_cls-resnet50_seed2026
python scripts/nnunet_e2e_inference.py --dataset-root $DATA --pred-dir $PRED --runs-dir runs
```

All runs use seed 2026. Confidence intervals in the paper are case-level bootstrap
intervals with 2,000 resamples.

## License

Apache License 2.0; see `LICENSE`.

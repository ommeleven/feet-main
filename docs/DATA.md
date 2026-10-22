# Data

The dataset is de-identified clinical ankle MRI with physician annotations. It was provided for research use and **is not distributed with this repository**. The `data/` directory is git-ignored.

## Contents (not public)

- MRI slices exported as JPEG, organized per patient and sequence (`P001 SAGIR`, `P001 SAGT1`, …). Sequences include SAGT1, SAGT2, SAGIR, AXT1, AXIR, AXPD, CORT1, CORIR and CORPD. The experiments use the sagittal SAGT1 and SAGIR series.
- Label Studio export (JSON) with rectangle (optionally rotated) bounding boxes labelled `Fracture` / `Edema`.
- Derived CSV tables of per-slice labels and box coordinates.

## Expected layout

Set `FOOT_FRACTURE_DATA_DIR` to a folder laid out like this (default: `./data`):

```
data/
├── raw/<PID> <SEQUENCE>/study/*.jpg         # original exported slices
├── json_data/sag_T1_json_columns/annotations/
│   └── results_box_cleaned_calculation.csv  # flattened Label Studio boxes
├── 864/
│   ├── train_st/{fracture,edema,healthy}/    # 3-way classifier
│   ├── test_st/{fracture,edema,healthy}/
│   ├── train_st_bin/{fracture,edema}/        # sequential step 2
│   ├── test_st_bin/{fracture,edema}/
│   └── sequential/healthy-unhealthy/{train,val}/{healthy,unhealthy}/   # step 1
└── *.csv                                     # intermediate annotation tables
```

The scripts in `footfracture/preprocessing/` and `footfracture/annotations/` build these folders from the raw export.

## Privacy

Do not commit images, annotation exports, or any file containing patient or study identifiers. The `.gitignore` blocks common image, archive and tabular formats as a safeguard. Still check `git status` before every commit.

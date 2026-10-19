# NIH Chest X-ray data (local)

Source: [NIH Chest X-rays on Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data)
(mirrored on HuggingFace `alkzar90/NIH-Chest-X-ray-dataset`; official Box links are flaky).

## Download

```bash
python scripts/download_nih_data.py --root data --start 1 --end 12
```

Then rebuild derived files:

```bash
python prepare_seg_data.py --csv data/Data_Entry_2017.csv --images data/raw --output data/seg_manifest.csv
python prepare_bbox_pseudomasks.py \
  --bbox-csv data/BBox_List_2017.csv --images data/raw \
  --mask-dir data/pseudo_masks --manifest data/bbox_pseudo_manifest.csv
```

`train_data.csv` / `test_data.csv` follow the official NIH `train_val_list.txt` / `test_list.txt`.

## Layout

- `Data_Entry_2017.csv`, `BBox_List_2017.csv`, split lists
- `raw/images/*.png` — 112,120 PNGs (1024×1024)
- `seg_manifest.csv` — patient-disjoint, initially `unannotated/pending`
- `bbox_pseudo_manifest.csv` — train-only weak bbox masks (`pseudo/pending`)
- `pseudo_masks/` — bbox rectangle masks

Segmentation still needs reviewed pixel masks before `train_seg.py`.

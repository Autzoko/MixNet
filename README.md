# MixNet

## Preprocess

**BUSI**
```bash
python data/preprocessor/busi.py --busi_root data/raw/BUSI --output_root data/processed/BUSI --val_ratio 0.2 --test_ratio 0.0
```

## Run

```bash
python scripts/train.py --data_root data/processed/BUSI/ --image_size 256 --batch_size 4 --epoch 100 --lr 1e-3 --device cuda
```
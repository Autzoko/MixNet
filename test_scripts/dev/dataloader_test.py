import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
import json
import numpy as np
import torch

from dataset.dataset_creators import create_contrastive_dataloaders, create_segmentation_dataloaders


if __name__ == "__main__":
    print("=" * 60)
    print("Ultrasound DataLoader Test (Mock Data)")
    print("=" * 60)
    
    import tempfile
    import os
    
    temp_dir = tempfile.mkdtemp()
    print(f"\nCreated temporary directory: {temp_dir}")
    
    num_samples = 10
    image_files = []
    mask_files = []
    
    for i in range(num_samples):
        img = np.random.rand(256, 256).astype(np.float32)
        img_path = os.path.join(temp_dir, f"image_{i:03d}.npy")
        np.save(img_path, img)
        image_files.append(img_path)
        
        mask = (np.random.rand(256, 256) > 0.5).astype(np.float32)
        mask_path = os.path.join(temp_dir, f"mask_{i:03d}.npy")
        np.save(mask_path, mask)
        mask_files.append(mask_path)
    
    # 修复：使用 'image_path' 和 'mask_path' 而不是 'image' 和 'mask'
    train_meta = []
    for i in range(8):
        train_meta.append({
            'id': f'sample_{i:03d}',
            'image_path': image_files[i],  # 修改：'image' -> 'image_path'
            'mask_path': mask_files[i],    # 修改：'mask' -> 'mask_path'
            'label': i % 2,
            'domain': 'BUSI'
        })
    
    val_meta = []
    for i in range(8, 10):
        val_meta.append({
            'id': f'sample_{i:03d}',
            'image_path': image_files[i],  # 修改：'image' -> 'image_path'
            'mask_path': mask_files[i],    # 修改：'mask' -> 'mask_path'
            'label': i % 2,
            'domain': 'BUSI'
        })
    
    train_meta_path = os.path.join(temp_dir, 'train_meta.json')
    val_meta_path = os.path.join(temp_dir, 'val_meta.json')
    
    with open(train_meta_path, 'w') as f:
        json.dump(train_meta, f)
    
    with open(val_meta_path, 'w') as f:
        json.dump(val_meta, f)
    
    print(f"\nCreated mock metadata files")
    
    print("\n" + "=" * 60)
    print("Test 1: Segmentation DataLoaders")
    print("=" * 60)
    
    train_loader, val_loader = create_segmentation_dataloaders(
        train_meta_path=train_meta_path,
        val_meta_path=val_meta_path,
        image_size=(256, 256),
        batch_size=4,
        num_workers=0,
        augment_train=True,
    )
    
    print(f"\nTrain loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    

    batch = next(iter(train_loader))
    print(f"\nTrain batch contents:")
    print(f"  id: {batch['id']}")
    print(f"  image: {batch['image'].shape}")
    print(f"  mask: {batch['mask'].shape}")
    print(f"  label: {batch['label'].shape if 'label' in batch else 'None'}")
    print(f"  domain: {batch['domain'] if 'domain' in batch else 'None'}")
    
    print("\n" + "=" * 60)
    print("Test 2: Contrastive Learning DataLoaders")
    print("=" * 60)
    
    contrastive_loaders = create_contrastive_dataloaders(
        source_meta_path=train_meta_path,
        target_meta_path=val_meta_path,
        image_size=(256, 256),
        batch_size=4,
        num_workers=0,
    )
    
    print(f"\nSource loader: {len(contrastive_loaders['source'])} batches")
    print(f"Target loader: {len(contrastive_loaders['target'])} batches")
    
    batch = next(iter(contrastive_loaders['source']))
    print(f"\nContrastive batch contents:")
    print(f"  id: {batch['id']}")
    print(f"  view1: {batch['view1'].shape}")
    print(f"  view2: {batch['view2'].shape}")
    print(f"  label: {batch['label'].shape if 'label' in batch else 'None'}")
    print(f"  domain: {batch['domain'] if 'domain' in batch else 'None'}")
    
    print("\n" + "=" * 60)
    print("Test 3: Data Range Verification")
    print("=" * 60)
    
    batch = next(iter(train_loader))
    image = batch['image']
    mask = batch['mask']
    
    print(f"\nImage statistics:")
    print(f"  min: {image.min().item():.4f}")
    print(f"  max: {image.max().item():.4f}")
    print(f"  mean: {image.mean().item():.4f}")
    print(f"  std: {image.std().item():.4f}")
    
    print(f"\nMask statistics:")
    print(f"  unique values: {torch.unique(mask).tolist()}")
    print(f"  positive ratio: {(mask > 0).float().mean().item():.4f}")
    
    print("\n" + "=" * 60)
    print("Cleaning up temporary files...")
    import shutil
    shutil.rmtree(temp_dir)
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
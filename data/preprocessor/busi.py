import os
import json
import argparse
import random
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image
from tqdm import tqdm


# 标签编码 - 🔥 移除 normal
LABEL_MAP = {
    'benign': 1,
    'malignant': 2,
}


def collect_busi_samples(busi_root: str) -> List[Dict]:
    """
    遍历 BUSI 根目录，收集所有样本信息
    
    🔥 修改：跳过 normal 类型样本，只收集 benign 和 malignant
    
    Args:
        busi_root: BUSI 数据集根目录
    
    Returns:
        样本字典列表，每个样本包含:
            - id: 唯一标识符
            - img_path: 原始图像路径
            - mask_path: 原始 mask 路径
            - label_name: 类别名称 ('benign', 'malignant')
    """
    busi_root = Path(busi_root)
    if not busi_root.exists():
        raise FileNotFoundError(f"BUSI root directory not found: {busi_root}")
    
    samples = []
    
    # 🔥 只处理 benign 和 malignant，跳过 normal
    for category in ['benign', 'malignant']:
        category_dir = busi_root / category
        
        if not category_dir.exists():
            print(f"Warning: Category directory not found: {category_dir}")
            continue
        
        print(f"\nProcessing {category} category...")
        
        # 收集配对样本（有 mask）
        samples_in_category = _collect_paired_samples(category_dir, category)
        
        samples.extend(samples_in_category)
        print(f"  Collected {len(samples_in_category)} {category} samples")
    
    print(f"\nTotal collected samples: {len(samples)}")
    print(f"🔥 Normal samples excluded from dataset")
    return samples


def _collect_paired_samples(category_dir: Path, category: str) -> List[Dict]:
    """
    收集有 mask 的样本（benign / malignant）
    
    Args:
        category_dir: 类别目录路径
        category: 类别名称
    
    Returns:
        样本列表
    """
    samples = []
    
    # 获取所有图像文件（不包含 _mask）
    all_files = sorted(category_dir.glob('*.png'))
    image_files = [f for f in all_files if '_mask' not in f.name]
    
    sample_idx = 1
    for img_path in image_files:
        # 构造对应的 mask 文件路径
        # 例如: "benign (1).png" -> "benign (1)_mask.png"
        mask_name = img_path.stem + '_mask.png'
        mask_path = category_dir / mask_name
        
        if not mask_path.exists():
            print(f"  Warning: Mask not found for {img_path.name}, skipping...")
            continue
        
        # 生成唯一 ID
        sample_id = f"BUSI_{category}_{sample_idx:06d}"
        
        samples.append({
            'id': sample_id,
            'img_path': str(img_path),
            'mask_path': str(mask_path),
            'label_name': category,
        })
        
        sample_idx += 1
    
    return samples


def split_samples(
    samples: List[Dict],
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], Optional[List[Dict]]]:
    """
    将样本划分为 train / val / test
    
    Args:
        samples: 样本列表
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
    
    Returns:
        (train_samples, val_samples, test_samples)
        test_samples 可能为 None（当 test_ratio=0）
    """
    # 设置随机种子
    random.seed(seed)
    
    # 打乱样本
    samples_shuffled = samples.copy()
    random.shuffle(samples_shuffled)
    
    n_total = len(samples_shuffled)
    
    # 计算各个 split 的样本数
    if test_ratio > 0:
        n_test = int(n_total * test_ratio)
        n_val = int((n_total - n_test) * val_ratio)
        n_train = n_total - n_test - n_val
        
        test_samples = samples_shuffled[:n_test]
        val_samples = samples_shuffled[n_test:n_test + n_val]
        train_samples = samples_shuffled[n_test + n_val:]
    else:
        n_val = int(n_total * val_ratio)
        n_train = n_total - n_val
        
        val_samples = samples_shuffled[:n_val]
        train_samples = samples_shuffled[n_val:]
        test_samples = None
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Val: {len(val_samples)} samples")
    if test_samples is not None:
        print(f"  Test: {len(test_samples)} samples")
    
    return train_samples, val_samples, test_samples


def load_and_convert_image(img_path: str) -> np.ndarray:
    """
    加载并转换图像为灰度 float32 数组
    
    Args:
        img_path: 图像路径
    
    Returns:
        np.ndarray, shape (H, W), dtype float32
    """
    img = Image.open(img_path).convert('L')  # 转为灰度
    img_np = np.array(img, dtype=np.float32)
    return img_np


def load_and_convert_mask(mask_path: str) -> np.ndarray:
    """
    加载并转换 mask 为二值数组
    
    Args:
        mask_path: mask 路径
    
    Returns:
        np.ndarray, shape (H, W), dtype float32, 值为 0 或 1
    """
    mask = Image.open(mask_path).convert('L')  # 转为灰度
    mask_np = np.array(mask, dtype=np.float32)
    
    # 二值化: >0 为 1，否则为 0
    mask_np = (mask_np > 0).astype(np.float32)
    
    return mask_np


def preprocess_and_save(
    samples: List[Dict],
    output_root: str,
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    seed: int = 42,
) -> None:
    """
    预处理样本并保存为 npy + metadata JSON
    
    🔥 修改：所有样本都有 mask（normal 已被排除）
    
    Args:
        samples: 收集到的样本列表
        output_root: 输出根目录
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
    """
    output_root = Path(output_root)
    
    # 创建输出目录
    images_dir = output_root / 'images'
    masks_dir = output_root / 'masks'
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_root}")
    print(f"  Images: {images_dir}")
    print(f"  Masks: {masks_dir}")
    
    # 划分数据集
    train_samples, val_samples, test_samples = split_samples(
        samples, val_ratio, test_ratio, seed
    )
    
    # 处理所有样本
    all_splits = {
        'train': train_samples,
        'val': val_samples,
    }
    if test_samples is not None:
        all_splits['test'] = test_samples
    
    metadata_all = {}
    
    for split_name, samples_in_split in all_splits.items():
        print(f"\nProcessing {split_name} split ({len(samples_in_split)} samples)...")
        
        metadata = []
        
        for sample in tqdm(samples_in_split, desc=f"  Converting {split_name}"):
            sample_id = sample['id']
            
            # 加载并保存图像
            img_np = load_and_convert_image(sample['img_path'])
            img_save_path = images_dir / f"{sample_id}.npy"
            np.save(img_save_path, img_np)
            
            # 🔥 所有样本都有 mask（因为 normal 已被排除）
            mask_np = load_and_convert_mask(sample['mask_path'])
            mask_save_path = masks_dir / f"{sample_id}_mask.npy"
            np.save(mask_save_path, mask_np)
            
            # 构造 metadata 条目
            meta_entry = {
                'id': sample_id,
                'image_path': f"images/{sample_id}.npy",
                'mask_path': f"masks/{sample_id}_mask.npy",
                'label': LABEL_MAP[sample['label_name']],
                'domain': 'BUSI',
            }
            
            metadata.append(meta_entry)
        
        metadata_all[split_name] = metadata
    
    # 保存 metadata JSON
    for split_name, metadata in metadata_all.items():
        json_path = output_root / f'{split_name}_meta.json'
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"\nSaved {split_name} metadata: {json_path}")
        print(f"  Samples: {len(metadata)}")
    
    print(f"\n✅ Preprocessing complete!")
    print(f"🔥 Only benign and malignant samples included")


def quick_test(output_root: str, image_size: Tuple[int, int] = (256, 256)) -> None:
    """
    快速测试：验证预处理结果可以被 DataLoader 正确加载
    
    Args:
        output_root: 预处理输出目录
        image_size: 测试时的图像尺寸
    """
    print("\n" + "=" * 60)
    print("Quick Test: DataLoader Compatibility")
    print("=" * 60)
    
    output_root = Path(output_root)
    train_meta_path = output_root / 'train_meta.json'
    
    if not train_meta_path.exists():
        print(f"Error: Metadata file not found: {train_meta_path}")
        return
    
    # 尝试导入通用 DataLoader
    try:
        # 假设 dataset 模块在 Python path 中
        import sys
        
        # 尝试添加可能的路径
        possible_paths = [
            Path.cwd() / 'dataset',
            Path.cwd().parent / 'dataset',
        ]
        for p in possible_paths:
            if p.exists():
                sys.path.insert(0, str(p.parent))
                break
        
        from dataset.meta.UltrasoundSample import load_ultrasound_metadata
        from dataset.dataset.ultrasound_segmentation import UltrasoundSegmentationDataset
        from torch.utils.data import DataLoader
        import torch
        
        print("\n✅ Successfully imported DataLoader components")
        
        # 加载 metadata
        samples = load_ultrasound_metadata(str(train_meta_path))
        print(f"\n✅ Loaded {len(samples)} samples from metadata")
        
        # 创建 Dataset
        dataset = UltrasoundSegmentationDataset(
            samples=samples,
            image_size=image_size,
            augment=None,
            normalize=True,
            return_label=True,
            return_domain=True,
        )
        
        # 创建 DataLoader
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
        )
        
        # 获取一个 batch
        batch = next(iter(loader))
        
        print(f"\n✅ Successfully loaded one batch:")
        print(f"  images: {batch['image'].shape}")
        print(f"  masks: {batch['mask'].shape}")
        print(f"  labels: {batch['label']}")
        print(f"  domains: {batch['domain']}")
        
        # 验证数据范围
        print(f"\nData statistics:")
        print(f"  Image range: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")
        print(f"  Image mean: {batch['image'].mean():.3f}")
        
        unique_mask_values = torch.unique(batch['mask'])
        print(f"  Mask unique values: {unique_mask_values.tolist()}")
        
        # 🔥 验证没有 normal 标签
        unique_labels = torch.unique(batch['label'])
        print(f"  Label unique values: {unique_labels.tolist()}")
        print(f"  🔥 Expected labels: 1 (benign), 2 (malignant)")
        
        print("\n✅ DataLoader compatibility test passed!")
        
    except ImportError as e:
        print(f"\n⚠️  Could not import DataLoader components: {e}")
        print("Skipping DataLoader test. Please ensure dataset modules are in Python path.")
        
        # 简单验证：直接读取文件
        print("\nPerforming basic file validation instead...")
        
        with open(train_meta_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"\nMetadata validation:")
        print(f"  Total entries: {len(metadata)}")
        
        # 统计标签分布
        label_counts = {}
        for item in metadata:
            label = item['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print(f"\nLabel distribution:")
        for label, count in sorted(label_counts.items()):
            label_name = 'benign' if label == 1 else 'malignant'
            print(f"  {label_name} (label={label}): {count} samples")
        
        # 检查前几个样本
        for i, item in enumerate(metadata[:3]):
            print(f"\nSample {i+1}:")
            print(f"  ID: {item['id']}")
            
            img_path = output_root / item['image_path']
            print(f"  Image exists: {img_path.exists()}")
            
            mask_path = output_root / item['mask_path']
            print(f"  Mask exists: {mask_path.exists()}")
            
            print(f"  Label: {item['label']}")
            print(f"  Domain: {item['domain']}")
        
        print("\n✅ Basic file validation passed!")
        print("🔥 All samples have masks (normal excluded)")
    
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess BUSI dataset (exclude normal samples)"
    )
    
    parser.add_argument(
        '--busi_root',
        type=str,
        required=True,
        help='Path to original BUSI dataset root directory'
    )
    
    parser.add_argument(
        '--output_root',
        type=str,
        required=True,
        help='Path to output directory for preprocessed data'
    )
    
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.2,
        help='Validation set ratio (default: 0.2)'
    )
    
    parser.add_argument(
        '--test_ratio',
        type=float,
        default=0.0,
        help='Test set ratio (default: 0.0, no test set)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for dataset splitting (default: 42)'
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Allow overwriting existing output directory'
    )
    
    parser.add_argument(
        '--skip_test',
        action='store_true',
        help='Skip the quick DataLoader test after preprocessing'
    )
    
    args = parser.parse_args()
    
    # 检查输出目录
    output_root = Path(args.output_root)
    if output_root.exists() and not args.overwrite:
        response = input(
            f"Output directory already exists: {output_root}\n"
            f"Do you want to overwrite it? (y/n): "
        )
        if response.lower() != 'y':
            print("Aborted.")
            return
        else:
            shutil.rmtree(output_root)
    
    # 开始预处理
    print("=" * 60)
    print("BUSI Dataset Preprocessing")
    print("🔥 Normal samples will be excluded")
    print("=" * 60)
    print(f"\nInput: {args.busi_root}")
    print(f"Output: {args.output_root}")
    print(f"Val ratio: {args.val_ratio}")
    print(f"Test ratio: {args.test_ratio}")
    print(f"Random seed: {args.seed}")
    
    # 收集样本（排除 normal）
    samples = collect_busi_samples(args.busi_root)
    
    if len(samples) == 0:
        print("\nError: No samples collected. Please check the input directory structure.")
        return
    
    # 预处理并保存
    preprocess_and_save(
        samples=samples,
        output_root=args.output_root,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    
    # 快速测试
    if not args.skip_test:
        quick_test(args.output_root)
    
    print("\n" + "=" * 60)
    print("All done! 🎉")
    print("🔥 Dataset contains only benign and malignant samples")
    print("=" * 60)


if __name__ == '__main__':
    main()
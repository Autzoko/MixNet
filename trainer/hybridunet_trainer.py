import os
import argparse
from pathlib import Path
import torch


from trainer.utils import set_seed, load_checkpoint, save_checkpoint
from trainer.utils.epoch import train_one_epoch, validate_one_epoch


def hybridunet_trainer():
    # ==================== 参数解析 ====================
    parser = argparse.ArgumentParser(
        description="Train HybridUNet on BUSI dataset"
    )
    
    parser.add_argument(
        '--data_root',
        type=str,
        default='BUSI_processed',
        help='Preprocessed BUSI data root directory'
    )
    
    parser.add_argument(
        '--image_size',
        type=int,
        default=256,
        help='Input image size (will be (size, size))'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Initial learning rate'
    )
    
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-4,
        help='Weight decay'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda/cpu)'
    )
    
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='DataLoader num_workers'
    )
    
    parser.add_argument(
        '--out_dir',
        type=str,
        default='outputs_busi_hybrid_unet',
        help='Output directory for checkpoints and logs'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    args = parser.parse_args()
    
    # ==================== 初始化 ====================
    print("=" * 60)
    print("HybridUNet Training on BUSI Dataset")
    print("=" * 60)
    
    # 设置随机种子
    set_seed(args.seed)
    print(f"\nRandom seed: {args.seed}")
    
    # 设置设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU instead")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")
    
    # 创建输出目录
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")
    
    # ==================== 数据加载 ====================
    print("\n" + "=" * 60)
    print("Loading Data")
    print("=" * 60)
    
    # 使用已有的 dataloader 创建函数
    try:
        from dataset.dataset_creators import create_segmentation_dataloaders
    except ImportError as e:
        raise ImportError(
            f"Failed to import create_segmentation_dataloaders: {e}\n"
            "Please ensure dataset modules are in Python path."
        )
    
    # 构建 metadata 路径
    train_meta_path = os.path.join(args.data_root, 'train_meta.json')
    val_meta_path = os.path.join(args.data_root, 'val_meta.json')
    
    train_loader, val_loader = create_segmentation_dataloaders(
        train_meta_path=train_meta_path,
        val_meta_path=val_meta_path,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment_train=True,  # 训练集启用增强
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # ==================== 模型构建 ====================
    print("\n" + "=" * 60)
    print("Building Model")
    print("=" * 60)
    
    try:
        from model.HybridUNet import HybridUNet
    except ImportError:
        raise ImportError(
            "Failed to import HybridUNet. "
            "Please ensure model modules are in Python path."
        )
    
    model = HybridUNet(
        in_ch=1,
        num_classes=1,
        base_ch=32,
        encoder_kwargs=None,
        decoder_kwargs=None,
    )
    model = model.to(device)
    
    # 打印模型信息
    model.print_model_info()
    
    # ==================== 优化器与调度器 ====================
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # 学习率调度器（CosineAnnealingLR）
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )
    
    # ==================== 加载 checkpoint（如果有）====================
    start_epoch = 1
    best_val_dice = 0.0
    
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer)
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_dice = checkpoint.get('best_val_dice', 0.0)
        
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # ==================== 训练循环 ====================
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)
    
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 60)
        
        # 训练
        train_loss, train_dice = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )
        
        # 验证
        val_loss, val_dice = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            device=device,
            epoch=epoch,
        )
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 打印统计
        print(f"\n  Train: loss={train_loss:.4f}, dice={train_dice:.4f}")
        print(f"  Val:   loss={val_loss:.4f}, dice={val_dice:.4f}")
        print(f"  LR: {current_lr:.2e}")
        
        # 保存 checkpoint
        is_best = val_dice > best_val_dice
        if is_best:
            best_val_dice = val_dice
            print(f"  ✨ New best val Dice: {best_val_dice:.4f}")
            
            checkpoint_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_dice': best_val_dice,
                'train_loss': train_loss,
                'train_dice': train_dice,
                'val_loss': val_loss,
                'val_dice': val_dice,
            }
            
            save_checkpoint(
                checkpoint_state,
                out_dir=args.out_dir,
                filename='best_model.pth',
            )
        else:
            print(f"  Best val Dice: {best_val_dice:.4f}")
        
        # 每 10 个 epoch 保存一次常规 checkpoint
        if epoch % 10 == 0:
            checkpoint_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_dice': best_val_dice,
            }
            save_checkpoint(
                checkpoint_state,
                out_dir=args.out_dir,
                filename=f'checkpoint_epoch_{epoch}.pth',
            )
    
    # ==================== 训练完成 ====================
    print("\n" + "=" * 60)
    print("Training Finished!")
    print("=" * 60)
    print(f"Best val Dice: {best_val_dice:.4f}")
    print(f"Best model saved to: {out_dir / 'best_model.pth'}")
    print("=" * 60)

import os
import argparse
from pathlib import Path
import torch


from trainer.utils import set_seed, load_checkpoint, save_checkpoint
from trainer.utils.epoch import train_one_epoch, validate_one_epoch

from trainer.loss.ultrasam import UltraSAMLoss


def hybridunet_trainer():
    # ==================== 参数解析 ====================
    parser = argparse.ArgumentParser(
        description="Train HybridUNet on BUSI dataset with UltraSAM Loss"
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
        default=1e-4,
        help='Initial learning rate (default: 1e-4 for UltraSAM Loss)'
    )
    
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-5,
        help='Weight decay (default: 1e-5)'
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
    
    # 🔥 UltraSAM Loss 参数
    parser.add_argument(
        '--lambda_dice',
        type=float,
        default=1.0,
        help='Weight for Dice loss (default: 1.0)'
    )
    
    parser.add_argument(
        '--lambda_focal',
        type=float,
        default=20.0,
        help='Weight for Focal loss (default: 20.0, increase for smaller lesions)'
    )
    
    parser.add_argument(
        '--lambda_iou',
        type=float,
        default=0.0,
        help='Weight for IoU loss (default: 0.0, set >0 if model has IoU head)'
    )
    
    parser.add_argument(
        '--focal_alpha',
        type=float,
        default=0.25,
        help='Focal loss alpha parameter (default: 0.25)'
    )
    
    parser.add_argument(
        '--focal_gamma',
        type=float,
        default=2.0,
        help='Focal loss gamma parameter (default: 2.0)'
    )
    
    parser.add_argument(
        '--max_grad_norm',
        type=float,
        default=1.0,
        help='Max gradient norm for clipping (default: 1.0, 0 to disable)'
    )
    
    args = parser.parse_args()
    
    # ==================== 初始化 ====================
    print("=" * 70)
    print("🚀 HybridUNet Training on BUSI Dataset")
    print("=" * 70)
    print()
    
    # 设置随机种子
    set_seed(args.seed)
    print(f"Random seed: {args.seed}")
    
    # 设置设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  Warning: CUDA not available, using CPU instead")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")
    
    # 创建输出目录
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")
    print()
    
    # ==================== 数据加载 ====================
    print("=" * 70)
    print("📊 Loading Data")
    print("=" * 70)
    
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
    
    print(f"Image size: {args.image_size} × {args.image_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print()
    
    # ==================== 模型构建 ====================
    print("=" * 70)
    print("🏗️  Building Model")
    print("=" * 70)
    
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
    print()
    
    # ==================== 损失函数 - UltraSAM Loss ====================
    print("=" * 70)
    print("🎯 Loss Function: UltraSAM Loss")
    print("=" * 70)
    
    criterion = UltraSAMLoss(
        lambda_dice=args.lambda_dice,
        lambda_focal=args.lambda_focal,
        lambda_iou=args.lambda_iou,
        dice_epsilon=1e-6,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
    ).to(device)
    
    print(f"Loss weights:")
    print(f"  λ_dice:  {args.lambda_dice:.2f}")
    print(f"  λ_focal: {args.lambda_focal:.2f}")
    print(f"  λ_iou:   {args.lambda_iou:.2f}")
    print(f"Focal parameters:")
    print(f"  alpha: {args.focal_alpha:.2f}")
    print(f"  gamma: {args.focal_gamma:.2f}")
    print()
    
    # ==================== 优化器与调度器 ====================
    print("=" * 70)
    print("⚙️  Optimizer & Scheduler")
    print("=" * 70)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    
    # 学习率调度器（CosineAnnealingLR）
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )
    
    print(f"Optimizer: AdamW")
    print(f"  Initial LR: {args.lr:.6f}")
    print(f"  Weight decay: {args.weight_decay:.6f}")
    print(f"  Betas: (0.9, 0.999)")
    print()
    print(f"Scheduler: CosineAnnealingLR")
    print(f"  T_max: {args.epochs}")
    print(f"  eta_min: 1e-6")
    print()
    print(f"Gradient clipping: {'Enabled' if args.max_grad_norm > 0 else 'Disabled'}")
    if args.max_grad_norm > 0:
        print(f"  Max norm: {args.max_grad_norm:.2f}")
    print()
    
    # ==================== 加载 checkpoint（如果有）====================
    start_epoch = 1
    best_val_dice = 0.0
    
    if args.resume:
        print("=" * 70)
        print("📥 Loading Checkpoint")
        print("=" * 70)
        
        checkpoint = load_checkpoint(args.resume, model, optimizer)
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_dice = checkpoint.get('best_val_dice', 0.0)
        
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Resumed from epoch {start_epoch - 1}")
        print(f"Best val Dice so far: {best_val_dice:.4f}")
        print()
    
    # ==================== 训练循环 ====================
    print("=" * 70)
    print("🎓 Starting Training")
    print("=" * 70)
    print(f"Epochs: {start_epoch} → {args.epochs}")
    print(f"Total epochs: {args.epochs}")
    print("=" * 70)
    print()
    
    for epoch in range(start_epoch, args.epochs + 1):
        # ============================================================
        # 训练
        # ============================================================
        train_loss, train_dice, train_components = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            max_grad_norm=args.max_grad_norm,
        )
        
        # ============================================================
        # 验证
        # ============================================================
        val_loss, val_dice, val_components = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
        )
        
        # ============================================================
        # 更新学习率
        # ============================================================
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # ============================================================
        # 打印详细统计
        # ============================================================
        print(f"\n{'='*70}")
        print(f"📈 Epoch [{epoch}/{args.epochs}] Summary")
        print(f"{'='*70}")
        
        print(f"\n🔹 Training Results:")
        print(f"  Total Loss:     {train_loss:.6f}")
        print(f"    ├─ Dice Loss:   {train_components['dice']:.6f}")
        print(f"    ├─ Focal Loss:  {train_components['focal']:.6f}")
        print(f"    ├─ IoU Loss:    {train_components['iou']:.6f}")
        print(f"    └─ Seg Loss:    {train_components['seg']:.6f}")
        print(f"  Dice Score:     {train_dice:.4f}")
        
        print(f"\n🔹 Validation Results:")
        print(f"  Total Loss:     {val_loss:.6f}")
        print(f"    ├─ Dice Loss:   {val_components['dice']:.6f}")
        print(f"    ├─ Focal Loss:  {val_components['focal']:.6f}")
        print(f"    ├─ IoU Loss:    {val_components['iou']:.6f}")
        print(f"    └─ Seg Loss:    {val_components['seg']:.6f}")
        print(f"  Dice Score:     {val_dice:.4f}")
        
        print(f"\n⚙️  Learning Rate: {current_lr:.6f}")
        
        # ============================================================
        # 保存 checkpoint
        # ============================================================
        is_best = val_dice > best_val_dice
        if is_best:
            best_val_dice = val_dice
            print(f"\n✅ New best val Dice: {best_val_dice:.4f}")
            
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
                'train_components': train_components,
                'val_components': val_components,
                'args': vars(args),  # 保存训练参数
            }
            
            save_checkpoint(
                checkpoint_state,
                out_dir=args.out_dir,
                filename='best_model.pth',
            )
            print(f"💾 Saved to: {out_dir / 'best_model.pth'}")
        else:
            print(f"\n   Best val Dice: {best_val_dice:.4f} (Epoch {epoch})")
        
        # 每 10 个 epoch 保存一次常规 checkpoint
        if epoch % 10 == 0:
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
                'args': vars(args),
            }
            save_checkpoint(
                checkpoint_state,
                out_dir=args.out_dir,
                filename=f'checkpoint_epoch_{epoch}.pth',
            )
            print(f"💾 Saved checkpoint: checkpoint_epoch_{epoch}.pth")
        
        print(f"{'='*70}\n")
    
    # ==================== 训练完成 ====================
    print("\n" + "=" * 70)
    print("🎉 Training Finished!")
    print("=" * 70)
    print(f"Best val Dice: {best_val_dice:.4f}")
    print(f"Best model saved to: {out_dir / 'best_model.pth'}")
    print("=" * 70)
    print()


if __name__ == '__main__':
    hybridunet_trainer()
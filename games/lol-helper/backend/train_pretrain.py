"""
预训练脚本
使用行为克隆训练模型
"""

import sys
import argparse
from pathlib import Path

import torch
import yaml

from utils import logger, PROJECT_ROOT, MODEL_DIR, CHECKPOINT_DIR
from ai.pretrain.model import LoLAIModel
from ai.pretrain.trainer import BehaviorCloningTrainer
from ai.pretrain.data_loader import (
    LoLDataset,
    create_sample_dataset
)


def custom_collate(batch):
    """自定义collate函数"""
    states, actions = zip(*batch)
    states = torch.stack(states, dim=0)
    actions = torch.stack(actions, dim=0)
    
    # 压缩action维度
    actions = actions.squeeze(dim=1)  # (batch,)
    
    # 重塑状态：(batch, 12*180*320) -> (batch, 12, 180, 320)
    states = states.reshape(states.size(0), 12, 180, 320)
    
    # 为状态添加序列维度
    states = states.unsqueeze(1)  # (batch, 1, 12, 180, 320)
    
    # 为actions添加序列维度
    actions = actions.unsqueeze(1)  # (batch, 1)
    
    return states, actions


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="英雄联盟AI预训练脚本")
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='backend/data/dataset/processed',
                       help='数据目录路径')
    parser.add_argument('--create_sample', action='store_true',
                       help='创建示例数据集')
    parser.add_argument('--sample_size', type=int, default=100,
                       help='示例数据集大小')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='验证集比例')
    
    # 模型参数
    parser.add_argument('--num_actions', type=int, default=32,
                       help='动作数量')
    parser.add_argument('--hidden_size', type=int, default=128,
                       help='LSTM隐藏层大小')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数')
    
    # 设备参数
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='训练设备')
    
    # 保存参数
    parser.add_argument('--save_dir', type=str, default='backend/ai/models',
                       help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='logs/runs',
                       help='TensorBoard日志目录')
    
    # 恢复训练
    parser.add_argument('--resume', type=str, default=None,
                       help='从检查点恢复训练')
    
    # 测试模式
    parser.add_argument('--test', action='store_true',
                       help='测试模式（仅1个epoch）')
    
    return parser.parse_args()


def main(args=None):
    """主函数"""
    if args is None:
        args = parse_args()
    
    print("=" * 60)
    print("英雄联盟AI预训练脚本")
    print("=" * 60)
    
    # 设置设备
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("\n[警告] CUDA不可用，使用CPU训练")
        device = 'cpu'
    
    print(f"\n[配置] 设备: {device}")
    print(f"[配置] 批次大小: {args.batch_size}")
    print(f"[配置] 学习率: {args.lr}")
    print(f"[配置] 训练轮数: {args.epochs}")
    if args.test:
        print(f"[配置] 测试模式: 仅1个epoch")
        args.epochs = 1
    
    # 创建示例数据集（如果需要）
    if args.create_sample:
        print(f"\n[数据] 创建示例数据集，大小: {args.sample_size}")
        sample_dir = Path(args.data_dir)
        sample_dir.mkdir(parents=True, exist_ok=True)
        create_sample_dataset(str(sample_dir / "sample_data.h5"), 
                            num_samples=args.sample_size,
                            num_actions=args.num_actions)
        print(f"[数据] 示例数据集已保存: {sample_dir / 'sample_data.h5'}")
    
    # 加载数据
    print(f"\n[数据] 加载数据集...")
    data_path = Path(args.data_dir)
    
    if data_path.exists() and (data_path / "sample_data.h5").exists():
        h5_file = str(data_path / "sample_data.h5")
    else:
        h5_file = str(data_path) if data_path.exists() else None
    
    if h5_file and Path(h5_file).exists():
        print(f"[数据] 数据文件: {h5_file}")
        
        # 创建数据集
        dataset = LoLDataset(h5_file)
        total_samples = len(dataset)
        print(f"[数据] 总样本数: {total_samples}")
        
        # 手动划分训练集和验证集
        train_size = int(total_samples * (1 - args.val_split))
        val_size = total_samples - train_size
        
        print(f"[数据] 训练集大小: {train_size}")
        print(f"[数据] 验证集大小: {val_size}")
        
        # 创建Subset
        from torch.utils.data import Subset
        if args.sample_size > 0 and args.sample_size < total_samples:
            train_size = int(args.sample_size * (1 - args.val_split))
            val_size = args.sample_size - train_size
            train_dataset = Subset(dataset, range(0, train_size))
            val_dataset = Subset(dataset, range(train_size, args.sample_size))
        else:
            train_dataset = Subset(dataset, range(0, train_size))
            val_dataset = Subset(dataset, range(train_size, total_samples))
        
        # 创建DataLoader
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
            collate_fn=custom_collate
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=True,
            collate_fn=custom_collate
        )
    else:
        print(f"[错误] 数据文件不存在: {h5_file}")
        print(f"[提示] 使用 --create_sample 创建示例数据集")
        sys.exit(1)
    
    # 创建模型
    print(f"\n[模型] 创建模型...")
    model = LoLAIModel(num_actions=args.num_actions, hidden_size=args.hidden_size)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[模型] 参数数量: {num_params:,}")
    
    # 创建训练器
    print(f"\n[训练] 创建训练器...")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    trainer = BehaviorCloningTrainer(
        model=model,
        train_dataloader=train_loader,
        lr=args.lr,
        device=device,
        log_dir=str(log_dir)
    )
    
    # 从检查点恢复（如果指定）
    if args.resume:
        print(f"\n[训练] 从检查点恢复: {args.resume}")
        trainer.load_model(args.resume)
    
    # 训练模型
    print(f"\n[训练] 开始训练...")
    history = trainer.train(
        num_epochs=args.epochs,
        val_dataloader=val_loader,
        save_dir=str(save_dir)
    )
    
    # 训练结果
    print(f"\n[结果] 训练完成")
    print(f"[结果] 最终训练损失: {history['train_loss'][-1]:.4f}")
    print(f"[结果] 最终训练准确率: {history['train_acc'][-1]:.4f}")
    if history['val_loss']:
        print(f"[结果] 最终验证损失: {history['val_loss'][-1]:.4f}")
        print(f"[结果] 最终验证准确率: {history['val_acc'][-1]:.4f}")
        print(f"[结果] 最佳验证损失: {min(history['val_loss']):.4f}")
    
    # 保存训练历史
    history_file = Path(args.save_dir) / "training_history.yaml"
    with open(history_file, 'w') as f:
        yaml.dump(history, f)
    print(f"[结果] 训练历史已保存: {history_file}")
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
小规模训练测试脚本
在Mac上快速验证训练流程（1-2个epoch）
"""

import sys
from pathlib import Path
import torch
import yaml

# 添加backend目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import logger, PROJECT_ROOT, MODEL_DIR, CHECKPOINT_DIR, LOGS_DIR
from ai.pretrain.model import LoLAIModel
from ai.pretrain.trainer import BehaviorCloningTrainer
from ai.pretrain.data_loader import LoLDataset, create_sample_dataset
from torch.utils.data import Subset, DataLoader


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


def main():
    print("=" * 60)
    print("小规模训练测试（Mac验证）")
    print("=" * 60)
    
    # 创建目录
    TEST_DIR = PROJECT_ROOT / "logs" / "mac_test"
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    
    # 创建示例数据集（小规模）
    print("\n[1/5] 创建示例数据集...")
    sample_path = TEST_DIR / "sample_data.h5"
    num_samples = 100
    num_actions = 32
    create_sample_dataset(str(sample_path), num_samples=num_samples)
    print(f"✓ 创建 {num_samples} 个样本")
    
    # 加载数据
    print("\n[2/5] 加载数据...")
    dataset = LoLDataset(str(sample_path))
    total_samples = len(dataset)
    print(f"✓ 总样本数: {total_samples}")
    
    # 划分训练集和验证集
    val_split = 0.2
    train_size = int(total_samples * (1 - val_split))
    val_size = total_samples - train_size
    
    train_dataset = Subset(dataset, range(0, train_size))
    val_dataset = Subset(dataset, range(train_size, total_samples))
    
    print(f"✓ 训练集: {train_size} 个样本")
    print(f"✓ 验证集: {val_size} 个样本")
    
    # 创建DataLoader
    batch_size = 8
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        collate_fn=custom_collate
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True,
        collate_fn=custom_collate
    )
    print(f"✓ DataLoader创建完成，batch_size={batch_size}")
    
    # 创建模型
    print("\n[3/5] 创建模型...")
    hidden_size = 64  # 小模型
    model = LoLAIModel(num_actions=num_actions, hidden_size=hidden_size)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ 模型参数: {num_params:,}")
    
    # 创建训练器
    print("\n[4/5] 创建训练器...")
    log_dir = TEST_DIR / "runs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    device = "cpu"  # Mac使用CPU
    lr = 0.001
    
    trainer = BehaviorCloningTrainer(
        model=model,
        train_dataloader=train_loader,
        lr=lr,
        device=device,
        log_dir=str(log_dir)
    )
    print(f"✓ 训练器创建完成，设备: {device}")
    
    # 训练
    print("\n[5/5] 开始训练（2个epoch）...")
    num_epochs = 2
    
    try:
        history = trainer.train(
            num_epochs=num_epochs,
            val_dataloader=val_loader,
            save_dir=str(TEST_DIR)
        )
        
        # 训练结果
        print("\n" + "=" * 60)
        print("训练完成！")
        print("=" * 60)
        print(f"\n最终训练损失: {history['train_loss'][-1]:.4f}")
        print(f"最终训练准确率: {history['train_acc'][-1]:.4f}")
        
        if history['val_loss']:
            print(f"最终验证损失: {history['val_loss'][-1]:.4f}")
            print(f"最终验证准确率: {history['val_acc'][-1]:.4f}")
            print(f"最佳验证损失: {min(history['val_loss']):.4f}")
        
        # 保存训练历史
        history_file = TEST_DIR / "training_history.yaml"
        with open(history_file, 'w') as f:
            yaml.dump(history, f)
        print(f"\n✓ 训练历史已保存: {history_file}")
        
        print("\n" + "=" * 60)
        print("✓ 所有测试通过！训练流程正常")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

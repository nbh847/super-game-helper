"""
训练器功能测试
验证模型和训练器是否能正常工作
"""

import sys
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# 添加backend目录到Python路径
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from utils import logger
from ai.pretrain.model import LoLAIModel
from ai.pretrain.trainer import BehaviorCloningTrainer


class MockDataset(Dataset):
    """模拟数据集"""
    
    def __init__(self, num_samples=100, seq_len=4, channels=12, 
                 height=180, width=320, num_actions=32):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.channels = channels
        self.height = height
        self.width = width
        self.num_actions = num_actions
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 随机生成状态
        state = torch.randn(self.channels, self.height, self.width)
        # 随机生成动作序列
        actions = torch.randint(0, self.num_actions, (self.seq_len,))
        return state, actions


def collate_fn(batch):
    """自定义collate函数，将序列数据堆叠"""
    states, actions = zip(*batch)
    
    # 堆叠状态和动作
    states = torch.stack(states, dim=0)
    actions = torch.stack(actions, dim=0)
    
    # 添加序列维度到状态
    states = states.unsqueeze(1).expand(-1, actions.size(1), -1, -1, -1)
    
    return states, actions


def test_model():
    """测试模型"""
    print("\n" + "=" * 60)
    print("测试1: 模型前向传播")
    print("=" * 60)
    
    # 创建模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    
    model = LoLAIModel(num_actions=32, hidden_size=128).to(device)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试前向传播
    batch_size = 4
    seq_len = 4
    channels = 12
    height = 180
    width = 320
    
    x = torch.randn(batch_size, seq_len, channels, height, width).to(device)
    logits, hidden = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {logits.shape}")
    print("✓ 模型前向传播测试通过")
    
    # 测试预测
    state = torch.randn(batch_size, channels, height, width).to(device)
    action_logits, hidden = model.predict(state)
    
    print(f"预测输入形状: {state.shape}")
    print(f"预测输出形状: {action_logits.shape}")
    print("✓ 模型预测测试通过")
    
    # 测试获取动作
    action, hidden = model.get_action(state)
    
    print(f"动作形状: {action.shape}")
    print(f"动作值: {action}")
    print("✓ 模型获取动作测试通过")
    
    return model


def test_dataloader():
    """测试数据加载器"""
    print("\n" + "=" * 60)
    print("测试2: 数据加载器")
    print("=" * 60)
    
    # 创建模拟数据集
    train_dataset = MockDataset(num_samples=100)
    val_dataset = MockDataset(num_samples=20)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # 测试数据加载
    for states, actions in train_loader:
        print(f"训练集批次形状:")
        print(f"  States: {states.shape}")
        print(f"  Actions: {actions.shape}")
        print(f"  Actions范围: [{actions.min()}, {actions.max()}]")
        break
    
    for states, actions in val_loader:
        print(f"验证集批次形状:")
        print(f"  States: {states.shape}")
        print(f"  Actions: {actions.shape}")
        break
    
    print("✓ 数据加载器测试通过")
    
    return train_loader, val_loader


def test_trainer(model, train_loader, val_loader):
    """测试训练器"""
    print("\n" + "=" * 60)
    print("测试3: 训练器")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建训练器
    trainer = BehaviorCloningTrainer(
        model=model,
        train_dataloader=train_loader,
        lr=0.001,
        device=device,
        log_dir="logs/test_runs"
    )
    
    # 测试训练一个epoch
    print("\n测试训练一个epoch...")
    train_loss, train_acc = trainer.train_epoch(epoch=1)
    print(f"训练损失: {train_loss:.4f}")
    print(f"训练准确率: {train_acc:.4f}")
    print("✓ 训练epoch测试通过")
    
    # 测试验证
    print("\n测试验证...")
    val_loss, val_acc = trainer.validate(val_loader)
    print(f"验证损失: {val_loss:.4f}")
    print(f"验证准确率: {val_acc:.4f}")
    print("✓ 验证测试通过")
    
    # 测试保存和加载模型
    print("\n测试保存和加载模型...")
    save_path = "logs/test_models/test_model.pth"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    trainer.save_model(save_path)
    print(f"模型已保存: {save_path}")
    
    trainer.load_model(save_path)
    print(f"模型已加载: {save_path}")
    print("✓ 模型保存和加载测试通过")
    
    return trainer


def test_training(trainer, train_loader, val_loader):
    """测试完整训练流程"""
    print("\n" + "=" * 60)
    print("测试4: 完整训练流程（2个epoch）")
    print("=" * 60)
    
    # 训练2个epoch
    history = trainer.train(
        num_epochs=2,
        val_dataloader=val_loader,
        save_dir="logs/test_models"
    )
    
    print("\n训练历史:")
    print(f"  训练损失: {history['train_loss']}")
    print(f"  训练准确率: {history['train_acc']}")
    print(f"  验证损失: {history['val_loss']}")
    print(f"  验证准确率: {history['val_acc']}")
    
    print("\n✓ 完整训练流程测试通过")
    
    return history


def main():
    """主函数"""
    print("=" * 60)
    print("训练器功能测试")
    print("=" * 60)
    
    try:
        # 测试1: 模型
        model = test_model()
        
        # 测试2: 数据加载器
        train_loader, val_loader = test_dataloader()
        
        # 测试3: 训练器
        trainer = test_trainer(model, train_loader, val_loader)
        
        # 测试4: 完整训练流程
        history = test_training(trainer, train_loader, val_loader)
        
        print("\n" + "=" * 60)
        print("✓ 所有测试通过！")
        print("=" * 60)
        
        print("\n总结:")
        print("  1. ✓ 模型前向传播正常")
        print("  2. ✓ 数据加载器正常")
        print("  3. ✓ 训练器训练和验证正常")
        print("  4. ✓ 模型保存和加载正常")
        print("  5. ✓ 完整训练流程正常")
        print("\n可以开始正式训练了！")
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

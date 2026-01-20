"""
训练器简化测试
"""

import sys
import torch
from pathlib import Path

backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from utils import logger
from ai.pretrain.model import LoLAIModel
from ai.pretrain.trainer import BehaviorCloningTrainer


class SimpleDataset(torch.utils.data.Dataset):
    """简单数据集"""
    def __init__(self, size=10):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # 返回: (state, actions)
        # state: (12, 180, 320)
        # actions: (4,)  # 4个时间步的动作
        state = torch.randn(12, 180, 320)
        actions = torch.randint(0, 32, (4,))
        return state, actions


def collate_fn(batch):
    """整理批次数据"""
    states, actions = zip(*batch)
    states = torch.stack(states, dim=0)  # (batch, 12, 180, 320)
    actions = torch.stack(actions, dim=0)  # (batch, 4)
    
    # 为状态添加序列维度
    states = states.unsqueeze(1).expand(-1, actions.size(1), -1, -1, -1)
    # (batch, 4, 12, 180, 320)
    
    return states, actions


def main():
    logger.info("开始简化测试...")
    
    # 创建数据集
    train_dataset = SimpleDataset(size=16)
    val_dataset = SimpleDataset(size=4)
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    logger.info("数据集和加载器创建完成")
    
    # 测试数据加载
    for states, actions in train_loader:
        logger.info(f"训练批次 - States: {states.shape}, Actions: {actions.shape}")
        break
    
    # 创建模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用设备: {device}")
    
    model = LoLAIModel(num_actions=32, hidden_size=128).to(device)
    logger.info(f"模型创建成功，参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建训练器
    logger.info("创建训练器...")
    trainer = BehaviorCloningTrainer(
        model=model,
        train_dataloader=train_loader,
        lr=0.001,
        device=device,
        log_dir="logs/test_runs"
    )
    
    # 测试训练一个epoch
    logger.info("测试训练一个epoch...")
    train_loss, train_acc = trainer.train_epoch(epoch=1)
    logger.info(f"训练完成 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
    
    # 测试验证
    logger.info("测试验证...")
    val_loss, val_acc = trainer.validate(val_loader)
    logger.info(f"验证完成 - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
    
    # 测试保存模型
    logger.info("测试保存模型...")
    save_path = "logs/test_models/test_model.pth"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    trainer.save_model(save_path)
    logger.info(f"模型已保存: {save_path}")
    
    logger.info("简化测试完成！")


if __name__ == "__main__":
    main()

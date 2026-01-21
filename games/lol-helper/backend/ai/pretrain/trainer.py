import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import time
import numpy as np

from .model import LoLAIModel


class BehaviorCloningTrainer:
    
    def __init__(self, model: LoLAIModel, train_dataloader, lr: float = 0.001, 
                 device: str = 'cuda', log_dir: Optional[str] = None):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.device = device
        
        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # TensorBoard日志
        self.log_dir = Path(log_dir) if log_dir else Path("logs/runs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
        print(f"[训练器] 初始化完成，设备: {self.device}")
        print(f"[训练器] 日志目录: {self.log_dir}")
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """训练一个epoch
        
        Returns:
            avg_loss: 平均损失
            avg_acc: 平均准确率
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        num_batches = len(self.train_dataloader)
        start_time = time.time()
        
        for batch_idx, (states, actions) in enumerate(self.train_dataloader):
            states = states.to(self.device)
            actions = actions.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            logits, _ = self.model(states)
            
            # 计算损失
            # logits: (batch, seq_len, num_actions)
            # actions: (batch, seq_len)
            batch_size, seq_len, num_actions = logits.shape
            logits_flat = logits.view(batch_size * seq_len, num_actions)
            actions_flat = actions.view(batch_size * seq_len)
            
            loss = self.criterion(logits_flat, actions_flat)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 优化器步骤
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(logits_flat, 1)
            total_correct += (predicted == actions_flat).sum().item()
            total_samples += actions_flat.size(0)
            
            # 打印进度
            if (batch_idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                batch_loss = total_loss / (batch_idx + 1)
                batch_acc = total_correct / total_samples
                print(f"  Epoch [{epoch}] [{batch_idx + 1}/{num_batches}] "
                      f"Loss: {batch_loss:.4f}, Acc: {batch_acc:.4f}, "
                      f"Time: {elapsed:.2f}s")
        
        avg_loss = total_loss / num_batches
        avg_acc = total_correct / total_samples
        
        return avg_loss, avg_acc
    
    def validate(self, val_dataloader) -> Tuple[float, float]:
        """验证模型
        
        Returns:
            avg_loss: 平均损失
            avg_acc: 平均准确率
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        num_batches = len(val_dataloader)
        
        with torch.no_grad():
            for batch_idx, (states, actions) in enumerate(val_dataloader):
                states = states.to(self.device)
                actions = actions.to(self.device)
                
                # 前向传播
                logits, _ = self.model(states)
                
                # 计算损失
                batch_size, seq_len, num_actions = logits.shape
                logits_flat = logits.view(batch_size * seq_len, num_actions)
                actions_flat = actions.view(batch_size * seq_len)
                
                loss = self.criterion(logits_flat, actions_flat)
                
                # 统计
                total_loss += loss.item()
                _, predicted = torch.max(logits_flat, 1)
                total_correct += (predicted == actions_flat).sum().item()
                total_samples += actions_flat.size(0)
        
        avg_loss = total_loss / num_batches
        avg_acc = total_correct / total_samples
        
        return avg_loss, avg_acc
    
    def train(self, num_epochs: int, val_dataloader = None, 
              save_dir: Optional[str] = None) -> Dict[str, Any]:
        """训练模型
        
        Args:
            num_epochs: 训练轮数
            val_dataloader: 验证数据加载器
            save_dir: 模型保存目录
        
        Returns:
            history: 训练历史
        """
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        save_dir = Path(save_dir) if save_dir else Path("backend/ai/models")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"开始训练，共 {num_epochs} 个epoch")
        print(f"{'='*60}")
        
        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            print(f"\nEpoch [{epoch}/{num_epochs}]")
            print("-" * 60)
            
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # 记录到TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            
            # 验证
            if val_dataloader is not None:
                val_loss, val_acc = self.validate(val_dataloader)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                # 记录到TensorBoard
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Accuracy/val', val_acc, epoch)
                
                # 学习率调度
                self.scheduler.step(val_loss)
                
                # 保存最佳模型
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    best_model_path = save_dir / "best_model.pth"
                    self.save_model(str(best_model_path))
                    print(f"\n  ✓ 最佳模型已保存: {best_model_path}")
                
                print(f"\n  训练结果 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
                print(f"  验证结果 - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            else:
                print(f"\n  训练结果 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            
            # 保存检查点
            if epoch % 5 == 0:
                checkpoint_path = save_dir / f"checkpoint_epoch_{epoch}.pth"
                self.save_checkpoint(str(checkpoint_path), epoch, history)
                print(f"  ✓ 检查点已保存: {checkpoint_path}")
            
            epoch_time = time.time() - epoch_start
            print(f"  Epoch耗时: {epoch_time:.2f}s")
        
        # 保存最终模型
        final_model_path = save_dir / "final_model.pth"
        self.save_model(str(final_model_path))
        print(f"\n{'='*60}")
        print(f"训练完成！")
        print(f"最终模型已保存: {final_model_path}")
        print(f"{'='*60}\n")
        
        # 关闭TensorBoard写入器
        self.writer.close()
        
        return history
    
    def save_model(self, path: str) -> None:
        """保存模型
        
        Args:
            path: 保存路径
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.current_epoch,
        }, path)
    
    def load_model(self, path: str) -> None:
        """加载模型
        
        Args:
            path: 模型路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        print(f"[训练器] 模型已加载: {path}")
    
    def save_checkpoint(self, path: str, epoch: int, 
                      history: Dict[str, Any]) -> None:
        """保存检查点
        
        Args:
            path: 保存路径
            epoch: 当前epoch
            history: 训练历史
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': history,
            'best_val_loss': self.best_val_loss,
        }, path)


if __name__ == "__main__":
    print("训练器测试...")
    
    from .data_loader import LoLDataset, get_dataloader
    from .model import LoLAIModel
    
    # 创建模拟数据
    print("\n创建模拟数据...")
    train_dataset = LoLDataset(num_samples=100)
    val_dataset = LoLDataset(num_samples=20)
    
    train_loader = get_dataloader(train_dataset, batch_size=4, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=4, shuffle=False)
    
    print(f"  训练集大小: {len(train_dataset)}")
    print(f"  验证集大小: {len(val_dataset)}")
    
    # 创建模型
    print("\n创建模型...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  使用设备: {device}")
    
    model = LoLAIModel(num_actions=32, hidden_size=128)
    print(f"  模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建训练器
    print("\n创建训练器...")
    trainer = BehaviorCloningTrainer(
        model=model,
        train_dataloader=train_loader,
        lr=0.001,
        device=device,
        log_dir="logs/test_runs"
    )
    
    # 训练（仅1个epoch测试）
    print("\n开始训练（测试1个epoch）...")
    history = trainer.train(num_epochs=1, val_dataloader=val_loader, save_dir="logs/test_models")
    
    print("\n训练历史:")
    print(f"  训练损失: {history['train_loss']}")
    print(f"  训练准确率: {history['train_acc']}")
    if val_loader:
        print(f"  验证损失: {history['val_loss']}")
        print(f"  验证准确率: {history['val_acc']}")
    
    print("\n训练器测试完成！")

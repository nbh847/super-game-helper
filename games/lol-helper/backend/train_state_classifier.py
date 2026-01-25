#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
英雄状态分类器训练脚本
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import argparse

import sys
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from ai.models.visual_encoder import VisualStateClassifier
from ai.models.hero_state_dataset import create_dataloaders, HeroStateDataset
from utils.paths import PROJECT_ROOT, MODEL_DIR
from utils.logger import logger


class TrainingConfig:
    """训练配置"""
    
    # 模型配置
    INPUT_CHANNELS = 3
    HIDDEN_DIM = 256
    NUM_CLASSES = 5
    DROPOUT = 0.3
    
    # 训练配置
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 50
    WEIGHT_DECAY = 1e-4
    
    # 学习率调度
    STEP_SIZE = 10
    GAMMA = 0.5
    
    # 设备配置
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 数据配置
    DATA_DIR = PROJECT_ROOT / "data" / "hero_states"
    TARGET_SIZE = (180, 320)  # (H, W)
    
    # 保存配置
    SAVE_DIR = MODEL_DIR / "state_classifier"
    SAVE_EPOCHS = 10  # 每N个epoch保存一次
    
    # TensorBoard配置
    LOG_DIR = PROJECT_ROOT / "logs" / "tensorboard" / "state_classifier"


class Trainer:
    """训练器"""
    
    def __init__(self, config, hero_names):
        self.config = config
        self.hero_names = hero_names
        self.device = config.DEVICE
        
        # 创建保存目录
        config.SAVE_DIR.mkdir(parents=True, exist_ok=True)
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        # 创建模型
        self.model = VisualStateClassifier(
            input_channels=config.INPUT_CHANNELS,
            hidden_dim=config.HIDDEN_DIM,
            num_classes=config.NUM_CLASSES,
            dropout=config.DROPOUT
        ).to(self.device)
        
        # 创建优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # 创建学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.STEP_SIZE,
            gamma=config.GAMMA
        )
        
        # 创建损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 创建TensorBoard
        self.writer = SummaryWriter(config.LOG_DIR)
        
        # 训练统计
        self.best_val_acc = 0.0
        self.global_step = 0
        
        logger.info(f"训练器初始化完成，设备: {self.device}")
        logger.info(f"英雄列表: {hero_names}")
    
    def train_epoch(self, train_loader, epoch):
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # TensorBoard记录
            self.writer.add_scalar('train/loss', loss.item(), self.global_step)
            self.global_step += 1
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        
        self.writer.add_scalar('train/epoch_loss', avg_loss, epoch)
        self.writer.add_scalar('train/epoch_acc', accuracy, epoch)
        
        return avg_loss, accuracy
    
    def validate(self, val_loader, epoch):
        """验证"""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="验证"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        
        self.writer.add_scalar('val/epoch_loss', avg_loss, epoch)
        self.writer.add_scalar('val/epoch_acc', accuracy, epoch)
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'config': self.config.__dict__
        }
        
        # 保存最新模型
        checkpoint_path = self.config.SAVE_DIR / "latest.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = self.config.SAVE_DIR / "best.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"保存最佳模型: {best_path}, 验证准确率: {val_acc:.2f}%")
        
        # 定期保存
        if epoch % self.config.SAVE_EPOCHS == 0:
            epoch_path = self.config.SAVE_DIR / f"epoch_{epoch}.pth"
            torch.save(checkpoint, epoch_path)
            logger.info(f"保存检查点: {epoch_path}")
    
    def update_training_status(self, val_acc):
        """更新训练状态到video_status.json"""
        from datetime import datetime
        
        video_status_path = self.config.DATA_DIR / "video_status.json"
        
        if not video_status_path.exists():
            logger.warning(f"video_status.json不存在: {video_status_path}")
            return
        
        # 读取当前状态
        with open(video_status_path, 'r', encoding='utf-8') as f:
            video_status = json.load(f)
        
        # 获取最佳模型路径（相对于项目根目录）
        best_model_path = self.config.SAVE_DIR / "best.pth"
        try:
            best_model_path = best_model_path.relative_to(PROJECT_ROOT)
        except ValueError:
            pass
        
        # 更新已训练视频的状态
        updated_count = 0
        for video_name, video_info in video_status.items():
            if video_info['hero_name'] in self.hero_names and video_info.get('status') == 'completed':
                if not video_info.get('trained', False):
                    video_info['trained'] = True
                    video_info['training_info'] = {
                        'model_path': str(best_model_path),
                        'training_date': datetime.now().isoformat(),
                        'val_acc': round(val_acc, 2)
                    }
                    updated_count += 1
        
        if updated_count > 0:
            # 保存更新后的状态
            with open(video_status_path, 'w', encoding='utf-8') as f:
                json.dump(video_status, f, ensure_ascii=False, indent=2)
            logger.info(f"更新训练状态: {updated_count}个视频标记为已训练")
        else:
            logger.info("没有需要更新训练状态的视频")
    
    def train(self, train_loader, val_loader, epochs):
        """训练"""
        logger.info("开始训练")
        
        final_val_acc = 0.0
        
        for epoch in range(1, epochs + 1):
            logger.info(f"Epoch {epoch}/{epochs}")
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # 验证
            val_loss, val_acc = self.validate(val_loader, epoch)
            final_val_acc = val_acc  # 保存最后一个epoch的验证准确率
            
            # 学习率调度
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('train/learning_rate', current_lr, epoch)
            
            # 日志
            logger.info(f"训练 - 损失: {train_loss:.4f}, 准确率: {train_acc:.2f}%")
            logger.info(f"验证 - 损失: {val_loss:.4f}, 准确率: {val_acc:.2f}%")
            logger.info(f"学习率: {current_lr:.6f}")
            
            # 保存模型
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
            
            self.save_checkpoint(epoch, val_acc, is_best)
        
        logger.info(f"训练完成，最佳验证准确率: {self.best_val_acc:.2f}%")
        self.writer.close()
        
        # 更新训练状态
        self.update_training_status(final_val_acc)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='英雄状态分类器训练')
    parser.add_argument('--heroes', type=str, nargs='+', required=True, help='英雄名称列表')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--resume', type=str, help='从检查点恢复训练')
    
    args = parser.parse_args()
    
    # 配置
    config = TrainingConfig()
    config.BATCH_SIZE = args.batch_size
    config.EPOCHS = args.epochs
    config.LEARNING_RATE = args.lr
    
    logger.info(f"训练配置: {config.__dict__}")
    
    # 创建数据加载器
    train_loader, val_loader, _ = create_dataloaders(
        config.DATA_DIR,
        args.heroes,
        batch_size=config.BATCH_SIZE,
        num_workers=2
    )
    
    # 创建训练器
    trainer = Trainer(config, args.heroes)
    
    # 恢复训练
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=config.DEVICE)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"从epoch {start_epoch}恢复训练")
    else:
        start_epoch = 1
    
    # 训练
    trainer.train(train_loader, val_loader, config.EPOCHS)


if __name__ == '__main__':
    main()

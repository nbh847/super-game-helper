#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
小型视觉编码器
用于从游戏画面提取特征，支持预训练和微调
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

import sys
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))

from utils.logger import logger


class VisualEncoder(nn.Module):
    """
    小型CNN视觉编码器
    
    输入: (B, 3, H, W) - RGB图像
    输出: (B, 256) - 特征向量
    
    参数量: ~0.5M
    """
    
    def __init__(self, input_channels=3, output_dim=256):
        super(VisualEncoder, self).__init__()
        
        self.input_channels = input_channels
        self.output_dim = output_dim
        
        # 卷积层
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # 自适应池化
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # 全连接层
        self.fc = nn.Linear(128 * 4 * 4, output_dim)
        
        # 初始化权重
        self._initialize_weights()
        
        logger.info(f"VisualEncoder初始化完成: 输入通道={input_channels}, 输出维度={output_dim}")
    
    def _initialize_weights(self):
        """Xavier初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: (B, 3, H, W) 输入图像
        
        Returns:
            features: (B, 256) 特征向量
        """
        # Conv1 + ReLU + BN
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Conv2 + ReLU + BN
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Conv3 + ReLU + BN
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # 自适应池化
        x = self.adaptive_pool(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接
        features = self.fc(x)
        
        return features
    
    def get_output_dim(self):
        """获取输出维度"""
        return self.output_dim
    
    def freeze(self):
        """冻结编码器参数"""
        for param in self.parameters():
            param.requires_grad = False
        logger.info("VisualEncoder已冻结")
    
    def unfreeze(self):
        """解冻编码器参数"""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("VisualEncoder已解冻")


class StateClassifier(nn.Module):
    """
    英雄状态分类器
    
    输入: (B, 256) - 视觉编码器输出
    输出: (B, 5) - 5种状态的概率
    
    状态: 移动、攻击、技能、受伤、死亡
    """
    
    def __init__(self, input_dim=256, num_classes=5, hidden_dim=128, dropout=0.3):
        super(StateClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # 隐藏层
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 输出层
        self.output = nn.Linear(hidden_dim, num_classes)
        
        # 初始化权重
        self._initialize_weights()
        
        logger.info(f"StateClassifier初始化完成: 输入维度={input_dim}, 类别数={num_classes}")
    
    def _initialize_weights(self):
        """Xavier初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, features):
        """
        前向传播
        
        Args:
            features: (B, 256) 特征向量
        
        Returns:
            logits: (B, 5) 各类别的logits
        """
        # 隐藏层
        x = self.hidden(features)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # 输出层
        logits = self.output(x)
        
        return logits
    
    def predict(self, features):
        """
        预测标签
        
        Args:
            features: (B, 256) 特征向量
        
        Returns:
            labels: (B,) 预测的标签
        """
        logits = self.forward(features)
        labels = torch.argmax(logits, dim=1)
        return labels


class VisualStateClassifier(nn.Module):
    """
    完整的视觉状态分类器
    
    组合视觉编码器和状态分类器
    """
    
    def __init__(self, input_channels=3, hidden_dim=256, num_classes=5, dropout=0.3):
        super(VisualStateClassifier, self).__init__()
        
        self.encoder = VisualEncoder(input_channels, hidden_dim)
        self.classifier = StateClassifier(hidden_dim, num_classes, hidden_dim, dropout)
        
        logger.info("VisualStateClassifier初始化完成")
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: (B, 3, H, W) 输入图像
        
        Returns:
            logits: (B, 5) 各类别的logits
        """
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits
    
    def predict(self, x):
        """
        预测标签
        
        Args:
            x: (B, 3, H, W) 输入图像
        
        Returns:
            labels: (B,) 预测的标签
        """
        logits = self.forward(x)
        labels = torch.argmax(logits, dim=1)
        return labels
    
    def freeze_encoder(self):
        """冻结编码器"""
        self.encoder.freeze()
    
    def unfreeze_encoder(self):
        """解冻编码器"""
        self.encoder.unfreeze()


def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_models():
    """测试模型"""
    print("=" * 50)
    print("测试视觉编码器")
    print("=" * 50)
    
    # 创建模型
    encoder = VisualEncoder(input_channels=3, output_dim=256)
    classifier = StateClassifier(input_dim=256, num_classes=5)
    full_model = VisualStateClassifier(input_channels=3, hidden_dim=256, num_classes=5)
    
    # 统计参数量
    print(f"VisualEncoder参数量: {count_parameters(encoder):,}")
    print(f"StateClassifier参数量: {count_parameters(classifier):,}")
    print(f"VisualStateClassifier参数量: {count_parameters(full_model):,}")
    
    # 测试前向传播
    batch_size = 4
    x = torch.randn(batch_size, 3, 180, 320)
    
    print(f"\n输入shape: {x.shape}")
    
    # 测试编码器
    features = encoder(x)
    print(f"编码器输出shape: {features.shape}")
    
    # 测试分类器
    logits = classifier(features)
    print(f"分类器输出shape: {logits.shape}")
    
    # 测试完整模型
    logits = full_model(x)
    labels = full_model.predict(x)
    print(f"完整模型输出shape: {logits.shape}")
    print(f"预测标签shape: {labels.shape}")
    
    # 测试冻结
    encoder.freeze()
    print("\n编码器已冻结")
    
    encoder.unfreeze()
    print("编码器已解冻")
    
    print("\n✅ 所有测试通过！")


if __name__ == '__main__':
    test_models()

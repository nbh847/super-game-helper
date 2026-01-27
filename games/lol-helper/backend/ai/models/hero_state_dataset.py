#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
英雄状态数据集
加载标注数据，支持数据增强
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
from torchvision import transforms

import sys
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))

from utils.logger import logger


class HeroStateDataset(Dataset):
    """
    英雄状态数据集
    
    支持数据增强和标签映射
    """
    
    LABEL_MAP = {
        '移动': 0,
        '攻击': 1,
        '技能': 2,
        '受伤': 3,
        '死亡': 4,
        '买装备': 5
    }
    
    REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
    
    def __init__(self, data_dir, hero_name, transform=None, target_size=(180, 320)):
        """
        Args:
            data_dir: 数据根目录
            hero_name: 英雄名称
            transform: 数据增强
            target_size: 目标图片尺寸 (H, W)
        """
        self.data_dir = Path(data_dir) / hero_name
        self.hero_name = hero_name
        self.target_size = target_size
        self.transform = transform
        
        # 检查数据结构：新结构（按视频） vs 旧结构（扁平）
        self.use_new_structure = self._check_data_structure()
        
        # 加载标签
        if self.use_new_structure:
            self._load_labels_new_structure()
        else:
            self._load_labels_old_structure()
        
        logger.info(f"加载数据集: {self.hero_name}, 有效样本数: {len(self.valid_samples)}, 结构: {'新' if self.use_new_structure else '旧'}")
    
    def _check_data_structure(self):
        """检查数据结构"""
        # 检查是否有视频目录（新结构）
        video_dirs = list(self.data_dir.glob("record*.mp4"))
        if video_dirs:
            return True
        
        # 检查是否有直接的frames目录（旧结构）
        if (self.data_dir / "frames").exists():
            return False
        
        # 如果都没有，尝试新结构
        return True
    
    def _load_labels_new_structure(self):
        """加载新结构数据（按视频组织）"""
        self.valid_samples = []
        
        # 遍历所有视频目录
        for video_dir in sorted(self.data_dir.glob("record*.mp4")):
            labels_file = video_dir / "labels.json"
            frames_dir = video_dir / "frames"
            
            if not labels_file.exists() or not frames_dir.exists():
                continue
            
            # 加载该视频的标签
            with open(labels_file, 'r', encoding='utf-8') as f:
                labels = json.load(f)
            
            # 收集有效样本
            for frame_name, label_name in labels.items():
                if label_name in self.LABEL_MAP:
                    frame_path = frames_dir / frame_name
                    if frame_path.exists():
                        self.valid_samples.append((str(frame_path), self.LABEL_MAP[label_name]))
    
    def _load_labels_old_structure(self):
        """加载旧结构数据（扁平结构）"""
        self.labels_file = self.data_dir / "labels.json"
        
        if not self.labels_file.exists():
            raise FileNotFoundError(f"标签文件不存在: {self.labels_file}")
        
        with open(self.labels_file, 'r', encoding='utf-8') as f:
            self.labels = json.load(f)
        
        # 过滤有效的标签
        self.valid_samples = []
        for frame_name, label_name in self.labels.items():
            if label_name in self.LABEL_MAP:
                frame_path = self.data_dir / "frames" / frame_name
                if frame_path.exists():
                    self.valid_samples.append((str(frame_path), self.LABEL_MAP[label_name]))
    
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        """
        获取样本
        
        Returns:
            image: (3, H, W) 图片张量
            label: 标签 (0-4)
        """
        frame_path, label = self.valid_samples[idx]
        
        # 读取图片
        image = cv2.imread(frame_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 调整大小
        image = cv2.resize(image, (self.target_size[1], self.target_size[0]))
        
        # 数据增强
        if self.transform:
            image = self.transform(image)
        else:
            # 默认转换
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return image, torch.tensor(label, dtype=torch.long)
    
    def get_class_distribution(self):
        """获取各类别的样本分布"""
        distribution = {name: 0 for name in self.LABEL_MAP.keys()}
        
        for _, label in self.valid_samples:
            label_name = self.REVERSE_LABEL_MAP[label]
            distribution[label_name] += 1
        
        return distribution


def get_transforms(train=True):
    """
    获取数据增强
    
    Args:
        train: 是否为训练集（训练集使用数据增强）
    
    Returns:
        transform: torchvision变换
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if train:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            normalize,
        ])


def create_dataloaders(data_dir, hero_names, batch_size=32, num_workers=2):
    """
    创建数据加载器
    
    Args:
        data_dir: 数据根目录
        hero_names: 英雄名称列表
        batch_size: 批次大小
        num_workers: 数据加载线程数
    
    Returns:
        train_loader, val_loader, test_loader
    """
    train_datasets = []
    val_datasets = []
    test_datasets = []
    
    for hero_name in hero_names:
        dataset = HeroStateDataset(data_dir, hero_name)
        
        # 划分数据集: 训练80% / 验证10% / 测试10%
        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size
        
        train_indices = list(range(0, train_size))
        val_indices = list(range(train_size, train_size + val_size))
        test_indices = list(range(train_size + val_size, total_size))
        
        # 创建子数据集
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)
        
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)
        test_datasets.append(test_dataset)
        
        logger.info(f"{hero_name}: 训练={len(train_indices)}, 验证={len(val_indices)}, 测试={len(test_indices)}")
    
    # 合并所有英雄的数据集
    from torch.utils.data import ConcatDataset
    
    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)
    test_dataset = ConcatDataset(test_datasets)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"数据加载器创建完成: 训练={len(train_dataset)}, 验证={len(val_dataset)}, 测试={len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


def test_dataset():
    """测试数据集"""
    print("=" * 50)
    print("测试英雄状态数据集")
    print("=" * 50)
    
    # 创建虚拟数据（用于测试）
    test_dir = Path(__file__).parent.parent.parent / "data" / "hero_states" / "test_dataset"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建标签文件
    labels = {
        "frame_0000.png": "移动",
        "frame_0001.png": "攻击",
        "frame_0002.png": "技能",
        "frame_0003.png": "受伤",
        "frame_0004.png": "死亡",
        "frame_0005.png": "移动",
    }
    
    labels_file = test_dir / "labels.json"
    with open(labels_file, 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    
    # 创建虚拟帧
    frames_dir = test_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(6):
        frame = np.random.randint(0, 255, (180, 320, 3), dtype=np.uint8)
        frame_path = frames_dir / f"frame_{i:04d}.png"
        cv2.imwrite(str(frame_path), frame)
    
    # 测试数据集
    dataset = HeroStateDataset(test_dir.parent, "test_dataset")
    
    print(f"数据集大小: {len(dataset)}")
    print(f"类别分布: {dataset.get_class_distribution()}")
    
    # 测试数据加载
    image, label = dataset[0]
    print(f"\n样本shape: image={image.shape}, label={label}")
    
    # 测试数据增强
    transform = get_transforms(train=True)
    dataset_with_transform = HeroStateDataset(test_dir.parent, "test_dataset", transform=transform)
    image, label = dataset_with_transform[0]
    print(f"增强后样本shape: {image.shape}, label={label}")
    
    print("\n✅ 数据集测试通过！")


if __name__ == '__main__':
    test_dataset()

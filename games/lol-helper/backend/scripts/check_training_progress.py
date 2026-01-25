#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import re
import os
from pathlib import Path

def check_model_checkpoints(save_dir):
    """检查模型检查点"""
    save_dir = Path(save_dir)
    
    if not save_dir.exists():
        print(f"模型保存目录不存在: {save_dir}")
        return []
    
    checkpoints = []
    
    # 查找所有.pth文件
    for pth_file in sorted(save_dir.glob("*.pth")):
        try:
            # 尝试加载检查点
            checkpoint = torch.load(pth_file, map_location='cpu')
            
            checkpoint_info = {
                'name': pth_file.name,
                'epoch': checkpoint.get('epoch', 0),
                'val_acc': checkpoint.get('val_acc', 0),
                'size': pth_file.stat().st_size / 1024 / 1024  # MB
            }
            
            checkpoints.append(checkpoint_info)
        except Exception as e:
            print(f"无法加载 {pth_file.name}: {e}")
    
    return checkpoints

def print_training_status(checkpoints):
    """打印训练状态"""
    if not checkpoints:
        print("没有找到训练检查点")
        print("\n可能的原因：")
        print("1. 训练尚未开始或未保存检查点")
        print("2. 模型保存目录路径错误")
        return

    print("\n" + "=" * 80)
    print("训练状态")
    print("=" * 80)
    print(f"{'文件名':<25} {'Epoch':<8} {'验证准确率':<12} {'文件大小':<12}")
    print("-" * 80)

    best_checkpoint = None
    best_val_acc = 0
    
    for cp in checkpoints:
        print(f"{cp['name']:<25} {cp['epoch']:<8} {cp['val_acc']:<12.2f}% {cp['size']:<12.2f} MB")
        
        if cp['val_acc'] > best_val_acc:
            best_val_acc = cp['val_acc']
            best_checkpoint = cp

    print("-" * 80)
    
    if best_checkpoint:
        print(f"最佳模型: {best_checkpoint['name']}")
        print(f"最佳验证准确率: {best_val_acc:.2f}%")
        print(f"训练轮数: {best_checkpoint['epoch']}")
    
    print("=" * 80)

if __name__ == '__main__':
    # 获取模型保存目录
    # 脚本在 backend/scripts/，所以需要回到上一级，再进入 ai/models/state_classifier
    script_path = Path(__file__).resolve()
    model_dir = script_path.parent.parent / "ai" / "models" / "state_classifier"
    
    # 检查检查点
    checkpoints = check_model_checkpoints(model_dir)
    print_training_status(checkpoints)

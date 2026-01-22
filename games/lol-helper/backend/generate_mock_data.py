#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
生成模拟测试数据
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image
import sys

backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from utils.paths import DATA_DIR
from utils.logger import logger


def generate_mock_dataset():
    """生成模拟数据集"""
    # 数据目录
    data_dir = DATA_DIR / "hero_states" / "test_dataset"
    frames_dir = data_dir / "frames"
    labels_file = data_dir / "labels.json"

    # 创建目录
    frames_dir.mkdir(parents=True, exist_ok=True)

    # 标签映射
    label_map = {
        "移动": 0,
        "攻击": 1,
        "技能": 2,
        "受伤": 3,
        "死亡": 4
    }

    # 生成标签和数据
    labels = {}
    num_samples = 100  # 生成100个样本

    # 每种状态生成20个样本
    samples_per_class = num_samples // len(label_map)

    for label_name, label_id in label_map.items():
        logger.info(f"生成{label_name}样本...")

        for i in range(samples_per_class):
            # 生成随机图像 (180x320 RGB)
            img_array = np.random.randint(0, 255, (180, 320, 3), dtype=np.uint8)

            # 根据标签添加一些特征
            if label_name == "移动":
                # 移动：添加一些纹理
                img_array[::10, ::10] = 255
            elif label_name == "攻击":
                # 攻击：红色区域
                img_array[70:110, 130:190] = [255, 0, 0]
            elif label_name == "技能":
                # 技能：蓝色区域
                img_array[70:110, 130:190] = [0, 0, 255]
            elif label_name == "受伤":
                # 受伤：低亮度
                img_array = (img_array * 0.5).astype(np.uint8)
            elif label_name == "死亡":
                # 死亡：灰色
                img_array = np.full_like(img_array, 100)

            # 保存图像
            frame_idx = label_id * samples_per_class + i
            frame_name = f"frame_{frame_idx:04d}.png"
            frame_path = frames_dir / frame_name

            img = Image.fromarray(img_array)
            img.save(frame_path)

            # 记录标签
            labels[frame_name] = label_name

    # 保存标签
    with open(labels_file, 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    logger.info(f"模拟数据集生成完成:")
    logger.info(f"  - 样本数: {num_samples}")
    logger.info(f"  - 图像目录: {frames_dir}")
    logger.info(f"  - 标签文件: {labels_file}")

    return data_dir


if __name__ == '__main__':
    generate_mock_dataset()

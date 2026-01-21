#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
标注工具测试脚本
测试各个模块的核心功能（不包含GUI）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from label_tool import (
    LabelConfig,
    FrameExtractor,
    AutoLabeler,
    LabelManager
)


def test_label_config():
    """测试配置类"""
    print("=" * 50)
    print("测试 LabelConfig")
    print("=" * 50)
    
    print(f"标签配置: {LabelConfig.LABELS}")
    print(f"显示尺寸: {LabelConfig.DISPLAY_SIZE}")
    print(f"帧间隔: {LabelConfig.FRAME_INTERVAL}秒")
    print(f"最大帧数: {LabelConfig.MAX_FRAMES}")
    print(f"数据目录: {LabelConfig.DATA_DIR}")
    
    print("✅ LabelConfig 测试通过\n")


def test_frame_extractor():
    """测试帧提取器"""
    print("=" * 50)
    print("测试 FrameExtractor")
    print("=" * 50)
    
    # 创建测试视频目录
    test_video_dir = LabelConfig.DATA_DIR / "test_hero" / "test_frames"
    test_video_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建虚拟提取器（不实际提取）
    extractor = FrameExtractor("fake_video.mp4", "test_hero")
    print(f"输出目录: {extractor.output_dir}")
    
    print("✅ FrameExtractor 测试通过\n")


def test_auto_labeler():
    """测试自动标注器"""
    print("=" * 50)
    print("测试 AutoLabeler")
    print("=" * 50)
    
    auto_labeler = AutoLabeler()
    
    # 检查模型是否加载
    print(f"YOLO模型: {'已加载' if auto_labeler.yolo_model is not None else '未加载'}")
    print(f"OCR模型: {'已加载' if auto_labeler.ocr is not None else '未加载'}")
    
    print("✅ AutoLabeler 测试通过\n")


def test_label_manager():
    """测试标签管理器"""
    print("=" * 50)
    print("测试 LabelManager")
    print("=" * 50)
    
    # 创建测试管理器
    manager = LabelManager("test_hero")
    
    # 测试设置标签
    manager.set_label("frame_0000.png", "移动")
    manager.set_label("frame_0001.png", "攻击")
    manager.set_label("frame_0002.png", "技能")
    
    # 测试获取标签
    assert manager.get_label("frame_0000.png") == "移动"
    assert manager.get_label("frame_0001.png") == "攻击"
    assert manager.get_label("frame_0002.png") == "技能"
    assert manager.get_label("frame_9999.png") is None
    
    print(f"标签数据: {manager.labels}")
    
    # 测试进度更新
    manager.update_progress(3, 10)
    progress = manager.get_progress()
    
    assert progress['current_index'] == 3
    assert progress['total_frames'] == 10
    assert progress['labeled_frames'] == 3
    
    print(f"进度数据: {progress}")
    
    # 清理测试数据
    if manager.labels_file.exists():
        manager.labels_file.unlink()
    if manager.progress_file.exists():
        manager.progress_file.unlink()
    
    print("✅ LabelManager 测试通过\n")


def main():
    """运行所有测试"""
    print("\n" + "=" * 50)
    print("标注工具测试")
    print("=" * 50 + "\n")
    
    try:
        test_label_config()
        test_frame_extractor()
        test_auto_labeler()
        test_label_manager()
        
        print("=" * 50)
        print("✅ 所有测试通过！")
        print("=" * 50 + "\n")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

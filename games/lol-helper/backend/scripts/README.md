# 标注工具目录

本目录包含英雄联盟大乱斗英雄状态标注工具。

## 文件说明

- `label_tool.py` - 标注工具主程序
- `test_label_tool.py` - 标注工具测试脚本
- `label_tool_user_guide.md` - 使用指南

## 快速开始

### 1. 运行标注工具

```bash
# 激活虚拟环境
source ../venv/bin/activate  # macOS/Linux
# 或
../venv/Scripts/activate  # Windows

# 从视频提取帧并标注
python label_tool.py --hero Lux --video data/hero_states/videos/Lux/game1.mp4

# 从已提取帧标注
python label_tool.py --hero Lux --frames-dir data/hero_states/Lux/frames/
```

### 2. 运行测试

```bash
python test_label_tool.py
```

## 功能特性

- ✅ 视频帧自动提取
- ✅ YOLO+OCR自动推断标签
- ✅ 人工确认界面（tkinter GUI）
- ✅ 标签数据管理（JSON存储）
- ✅ 进度保存和恢复
- ✅ 快捷键支持

## 标注类型

- `1` - 移动（绿色）
- `2` - 攻击（红色）
- `3` - 技能（蓝色）
- `4` - 受伤（黄色）
- `5` - 死亡（灰色）

## 快捷键

- `1-5` - 设置标签
- `←/→` - 上一帧/下一帧
- `S` - 保存数据
- `Q` - 退出工具

## 详细文档

请参阅 [label_tool_user_guide.md](label_tool_user_guide.md) 获取详细使用说明。

## 输出数据

标注完成后，数据保存在 `data/hero_states/{英雄名称}/` 目录下：

- `frames/` - 提取的帧图片
- `labels.json` - 标签数据
- `progress.json` - 进度数据

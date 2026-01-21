# 英雄状态标注工具使用指南

## 概述

英雄状态标注工具用于标注英雄联盟大乱斗游戏中英雄的状态，支持YOLO+OCR自动推断和人工确认。

标注工具支持5种状态：
- **移动**：英雄正在移动
- **攻击**：英雄正在攻击敌人或小兵
- **技能**：英雄正在释放技能
- **受伤**：英雄受到伤害
- **死亡**：英雄死亡

## 安装依赖

### 1. 基础依赖

标注工具需要以下Python依赖：

```bash
# 激活虚拟环境
cd backend
source ../venv/bin/activate  # macOS/Linux
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 图像处理依赖

标注工具使用OpenCV处理图像：

```bash
pip install opencv-python
```

### 3. GUI依赖

标注工具使用tkinter（Python内置，无需安装）：

```bash
# 验证tkinter是否可用
python -c "import tkinter; print('tkinter可用')"
```

### 4. YOLO和OCR（可选，用于自动推断）

如果需要使用自动推断功能，需要安装以下依赖：

```bash
# YOLOv8
pip install ultralytics

# PaddleOCR
pip install paddleocr
```

**注意**：首次运行时，YOLO和OCR模型会自动下载，需要联网。

## 标注流程

### 步骤1：准备视频数据

将游戏录像视频（.mp4, .avi, .mkv）放到合适的位置：

- 单个视频：`data/hero_states/videos/{hero_name}/game1.mp4`
- 多个视频：`data/hero_states/videos/{hero_name}/`

### 步骤2：运行标注工具

**从视频提取帧并标注**：

```bash
python scripts/label_tool.py --hero <英雄名称> --video <视频路径>
```

示例：
```bash
# 单个视频
python scripts/label_tool.py --hero Lux --video data/hero_states/videos/Lux/game1.mp4

# 视频目录
python scripts/label_tool.py --hero Lux --video data/hero_states/videos/Lux/
```

**从已提取帧标注**：

```bash
python scripts/label_tool.py --hero <英雄名称> --frames-dir <帧目录>
```

示例：
```bash
python scripts/label_tool.py --hero Lux --frames-dir data/hero_states/Lux/frames/
```

### 步骤3：使用工具标注

工具启动后会显示GUI界面，包含：
- 图片显示区域（显示当前帧）
- 当前标签显示
- AI建议标签（如果启用自动推断）
- 快捷键提示
- 按钮栏（上一帧、下一帧、保存、退出）

### 步骤4：保存标注数据

标注过程中，数据会自动保存。也可以手动点击"保存"按钮或按`S`键保存。

标注完成后，数据保存在：
- 帧图片：`data/hero_states/{英雄名称}/frames/`
- 标签数据：`data/hero_states/{英雄名称}/labels.json`
- 进度数据：`data/hero_states/{英雄名称}/progress.json`

## 快捷键说明

| 快捷键 | 功能 |
|--------|------|
| `1` | 标注为"移动" |
| `2` | 标注为"攻击" |
| `3` | 标注为"技能" |
| `4` | 标注为"受伤" |
| `5` | 标注为"死亡" |
| `←` | 上一帧 |
| `→` | 下一帧 |
| `S` | 保存数据 |
| `Q` | 退出工具 |

## 数据格式

### labels.json

```json
{
  "frame_0000.png": "移动",
  "frame_0001.png": "攻击",
  "frame_0002.png": "技能",
  "frame_0003.png": "死亡",
  "frame_0004.png": "移动"
}
```

### progress.json

```json
{
  "current_index": 4,
  "total_frames": 200,
  "labeled_frames": 5,
  "last_updated": "2026-01-21T16:00:00"
}
```

## 配置选项

标注工具支持以下配置选项（在代码中修改）：

- `LabelConfig.FRAME_INTERVAL`：帧提取间隔（秒），默认1秒
- `LabelConfig.MAX_FRAMES`：最大提取帧数，默认200帧
- `LabelConfig.DISPLAY_SIZE`：显示图片尺寸，默认(800, 600)

## 使用建议

### 1. 数据收集建议

- 每个英雄收集10-20局游戏录像
- 每局录像约200帧（每秒1帧）
- 确保覆盖各种游戏场景（战斗、移动、死亡等）

### 2. 标注建议

- 先快速浏览所有帧，了解整体情况
- 使用AI建议作为参考，人工确认
- 遇到不确定的帧，可以跳过（不标注）
- 定期保存数据（按S键）

### 3. 质量控制

- 确保标签准确性
- 标注完成后，随机抽查10%的帧
- 发现错误及时修正

## 常见问题解答

### Q1: 提示"未找到任何帧"

**A**: 检查以下内容：
- 视频路径是否正确
- 视频格式是否支持（.mp4, .avi, .mkv）
- 视频文件是否损坏

### Q2: GUI界面显示异常

**A**: 
- 确保tkinter已安装：`python -c "import tkinter"`
- 尝试调整屏幕分辨率
- 检查图片尺寸配置（DISPLAY_SIZE）

### Q3: AI推断功能不工作

**A**:
- 确保已安装YOLO和OCR依赖
- 检查网络连接（首次运行需要下载模型）
- 查看日志文件确认模型加载状态

### Q4: 数据保存失败

**A**:
- 检查磁盘空间
- 检查文件权限
- 确保目录路径正确

### Q5: 如何继续之前的标注？

**A**:
- 使用`--frames-dir`参数指定已有的帧目录
- 工具会自动加载之前的标注数据
- 从上次中断的地方继续

### Q6: 标注数据可以用在哪里？

**A**:
- 训练预训练视觉编码器
- 训练英雄状态分类器
- 评估AI模型性能

## 技术细节

### 自动推断算法

标注工具使用YOLOv8n进行目标检测，PaddleOCR进行文字识别。

**目标检测**：
- 英雄检测
- 小兵检测
- 防御塔检测

**文字识别**：
- 血条数值
- 金币数量

**启发式推断**：
- 根据检测结果和游戏状态推断标签
- 默认标签为"移动"
- 如果检测到大量目标，推断为"攻击"

### 性能优化

- 帧提取：使用OpenCV高效提取
- GUI渲染：使用tkinter轻量级GUI
- 数据存储：JSON格式，易于读写

## 后续步骤

标注完成后，可以进行：

1. **数据验证**：检查标注质量
2. **数据增强**：增加样本多样性
3. **模型训练**：使用标注数据训练模型
4. **模型评估**：评估模型性能

## 联系支持

如有问题，请查阅：
- 项目文档：`docs/`
- API参考：`docs/api_reference.md`
- 进度记录：`docs/progress.md`

---

**文档版本**: 1.0  
**创建日期**: 2026-01-21  
**最后更新**: 2026-01-21

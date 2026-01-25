# 英雄状态数据结构说明

## 数据组织方式

英雄状态数据按**视频**组织，便于追踪和管理每个对局的标注数据。

## 目录结构

```
data/hero_states/
├── video_status.json              # 视频标注状态（全局）
│
├── {英雄名称}/                   # 英雄数据目录
│   ├── progress.json            # 标注进度
│   │
│   ├── {视频文件名}/           # 视频数据目录（每个对局一个）
│   │   ├── frames/             # 帧图片
│   │   │   ├── frame_0000.png
│   │   │   ├── frame_0001.png
│   │   │   └── ...
│   │   └── labels.json         # 标签数据
│   │
│   ├── old_frames/             # 旧数据（迁移后）
│   └── old_labels.json         # 旧标签（迁移后）
│
└── {英雄名称}_backup_{时间戳}/   # 备份目录（迁移前）
```

## 文件说明

### 1. video_status.json（全局）

**位置**：`data/hero_states/video_status.json`

**作用**：记录所有视频的标注和训练状态

**字段说明**：
```json
{
  "record20260123-222005-塞拉斯-win.mp4": {
    "video_path": "E:\\MyGame\\GameVideos\\...\\record20260123-222005-塞拉斯-win.mp4",
    "hero_name": "塞拉斯",
    "status": "completed",  // pending/labeling/completed
    "total_frames": 290,
    "labeled_frames": 290,
    "strategy_version": "v1.0",
    "history": [],
    "created_at": "2026-01-24T13:11:10",
    "completed_at": "2026-01-25T10:26:54",
    "trained": false,  // 是否已用于训练
    "training_info": {  // 训练信息（训练后）
      "model_path": "backend/ai/models/state_classifier/best.pth",
      "training_date": "2026-01-25T14:00:00",
      "val_acc": 78.5
    }
  }
}
```

### 2. labels.json（视频级别）

**位置**：`data/hero_states/{英雄名称}/{视频文件名}/labels.json`

**作用**：存储该视频的帧标签

**格式**：
```json
{
  "frame_0000.png": "移动",
  "frame_0001.png": "攻击",
  "frame_0002.png": "技能",
  "frame_0003.png": "受伤",
  "frame_0004.png": "死亡"
}
```

### 3. progress.json（英雄级别）

**位置**：`data/hero_states/{英雄名称}/progress.json`

**作用**：记录标注进度

**格式**：
```json
{
  "current_index": 289,
  "total_frames": 300,
  "labeled_frames": 290,
  "last_updated": "2026-01-25T13:42:00"
}
```

## 数据迁移

### 旧结构 vs 新结构

**旧结构（扁平）：**
```
data/hero_states/塞拉斯/
├── frames/           # 所有视频的帧混合
│   ├── frame_0000.png
│   └── frame_0001.png
├── labels.json       # 所有标签混合
└── progress.json
```

**新结构（按视频）：**
```
data/hero_states/塞拉斯/
├── record20260123-222005-塞拉斯-win.mp4/
│   ├── frames/
│   └── labels.json
└── progress.json
```

### 迁移操作

如果需要从旧结构迁移到新结构：

```bash
# 运行迁移脚本
backend/venv/Scripts/python.exe backend/scripts/migrate_hero_data.py
```

迁移脚本会：
1. 备份旧数据到 `{英雄名}_backup_{时间戳}/`
2. 为每个视频创建独立目录
3. 分配帧和标签到对应的视频目录
4. 旧数据重命名为 `old_*`

### 验证迁移

```bash
# 检查每个英雄的视频目录数量
for hero in 塞拉斯 蒙多 蔚 瑞兹 梅尔; do
  ls -d "data/hero_states/$hero/"record*.mp4 2>/dev/null | wc -l
done

# 检查每个视频的帧数
ls "data/hero_states/塞拉斯/record20260123-222005-塞拉斯-win.mp4/frames/" | wc -l

# 检查labels.json
cat "data/hero_states/梅尔/record20260124-161447.mp4/labels.json" | python -m json.tool
```

### 清理旧数据

确认迁移无误后，可以删除旧数据：

```bash
# 删除旧数据
rm -rf data/hero_states/*/old_*

# 删除备份
rm -rf data/hero_states/*_backup_*
```

## 训练数据使用

### 按英雄训练

```bash
backend/venv/Scripts/python.exe backend/train_state_classifier.py \
  --heroes 塞拉斯 蒙多 蔚 瑞兹 梅尔 \
  --batch-size 16 \
  --epochs 50 \
  --lr 0.001
```

### 增量训练（基于已有模型）

```bash
backend/venv/Scripts/python.exe backend/train_state_classifier.py \
  --heroes 塞拉斯 蒙多 蔚 瑞兹 梅尔 新英雄 \
  --batch-size 16 \
  --epochs 50 \
  --lr 0.001 \
  --resume backend/ai/models/state_classifier/best.pth
```

### 训练策略

1. **完全重训**（推荐）：每标注完5个新英雄，用全部数据从头训练
2. **增量训练**：每标注1-2个新英雄，基于已有模型继续训练
3. **定期校准**：每10个新英雄，做一次完全重训校准模型

## 数据统计

### 统计已标注的英雄

```bash
backend/venv/Scripts/python.exe backend/scripts/label_tool.py --status
```

### 统计总帧数

```bash
find data/hero_states -name "labels.json" -exec wc -l {} \; | awk '{sum+=$1} END {print "总帧数:", sum}'
```

### 统计每个英雄的数据分布

```python
import json
from pathlib import Path

for hero_dir in Path("data/hero_states").glob("*/"):
    if not hero_dir.is_dir():
        continue
    
    print(f"\n=== {hero_dir.name} ===")
    
    for video_dir in hero_dir.glob("record*.mp4/"):
        labels_file = video_dir / "labels.json"
        if labels_file.exists():
            with open(labels_file, 'r', encoding='utf-8') as f:
                labels = json.load(f)
            print(f"{video_dir.name}: {len(labels)} 帧")
```

## 常见问题

### Q1: 如何知道哪些视频已经训练过？

**A**: 查看 `video_status.json` 中的 `trained` 字段：
```json
{
  "trained": true,
  "training_info": {
    "model_path": "backend/ai/models/state_classifier/best.pth",
    "training_date": "2026-01-25T14:00:00",
    "val_acc": 78.5
  }
}
```

### Q2: 同一英雄有多个对局，数据会混乱吗？

**A**: 不会。每个对局（视频）都有独立的目录：
```
塞拉斯/
├── record20260123-222005-塞拉斯-win.mp4/
└── record20260126-080000-塞拉斯-win.mp4/
```

### Q3: 如何删除某个视频的标注？

**A**: 删除对应的视频目录：
```bash
rm -rf "data/hero_states/塞拉斯/record20260123-222005-塞拉斯-win.mp4/"
```

### Q4: 如何重新标注某个视频？

**A**:
1. 删除视频目录（如上）
2. 运行标注工具：
```bash
backend/venv/Scripts/python.exe backend/scripts/label_tool.py \
  --hero 塞拉斯 \
  --video "path/to/video.mp4"
```

### Q5: 训练时如何包含所有英雄？

**A**: 列出所有英雄名称：
```bash
--heroes 塞拉斯 蒙多 蔚 瑞兹 梅尔 英雄6 英雄7 ...
```

## 版本历史

- **v1.0** (2026-01-25): 从扁平结构迁移到按视频组织
- **v0.x** (2026-01-21 - 2026-01-24): 扁平结构（已废弃）

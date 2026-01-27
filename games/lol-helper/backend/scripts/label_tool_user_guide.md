# 英雄状态标注工具使用指南

## 概述

英雄状态标注工具用于标注英雄联盟大乱斗游戏中英雄的状态，支持YOLO+OCR自动推断和人工确认。

标注工具支持6种状态：
- **移动**：英雄正在移动
- **攻击**：英雄正在攻击敌人或小兵
- **技能**：英雄正在释放技能
- **受伤**：英雄受到伤害
- **死亡**：英雄死亡
- **买装备**：英雄在商店购买装备

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
- 帧图片：`data/hero_states/{英雄名称}/{视频文件名}/frames/`
- 标签数据：`data/hero_states/{英雄名称}/{视频文件名}/labels.json`
- 进度数据：`data/hero_states/{英雄名称}/progress.json`

**数据结构示例：**
```
data/hero_states/
├── 塞拉斯/
│   ├── record20260123-222005-塞拉斯-win.mp4/
│   │   ├── frames/
│   │   │   ├── frame_0000.png
│   │   │   └── frame_0001.png
│   │   └── labels.json
│   └── progress.json
├── 梅尔/
│   ├── record20260124-161447.mp4/
│   │   ├── frames/
│   │   └── labels.json
│   └── progress.json
```

## 快捷键说明

| 快捷键 | 功能 |
|--------|------|
| `1` | 标注为"移动" |
| `2` | 标注为"攻击" |
| `3` | 标注为"技能" |
| `4` | 标注为"受伤" |
| `5` | 标注为"死亡" |
| `6` | 标注为"买装备" |
| `←` | 上一帧 |
| `→` | 下一帧 |
| `S` | 保存数据 |
| `Q` | 退出工具 |

## 视频标注状态跟踪

标注工具会自动跟踪每个视频的标注状态，避免重复标注已完成的视频。

### 查看视频状态

```bash
python scripts/label_tool.py --status
```

输出示例：
```
================================================================================
视频标注状态 (当前策略版本: v1.0)
================================================================================
✅ game1.webm (Lux)
   状态: completed | 进度: 200/200 (100%)
   完成时间: 2026-01-23T16:00:00
```

### 🎮 游戏资料管理

项目中存在**两个独立的状态文件**，作用不同：

#### 1. video_status.json - 视频标注状态

**位置**：`data/hero_states/video_status.json`

**作用**：记录视频标注进度，避免重复标注

**字段说明**：
```json
{
  "record_abc123...": {
    "video_path": "E:\\MyGame\\GameVideos\\lolhighlight\\...",
    "hero_name": "塞拉斯",  // 翻译名
    "status": "completed",  // pending/labeling/completed
    "total_frames": 300,      // 总帧数
    "labeled_frames": 300,    // 已标注帧数
    "strategy_version": "v1.0", // 标注策略版本
    "history": [],                 // 标注历史
    "created_at": "2026-01-24T13:11:10",  // 创建时间
    "completed_at": null,          // 完成时间
    "trained": false,  // 是否已用于训练
    "training_info": {  // 训练信息（训练后自动添加）
      "model_path": "backend/ai/models/state_classifier/best.pth",
      "training_date": "2026-01-25T14:00:00",
      "val_acc": 72.22
    }
  }
}
```

#### 2. record_details.json - 游戏对局资料

**位置**：`E:\MyGame\GameVideos\record_details.json`

**作用**：保存游戏对局的详细资料，方便标注时快速了解视频信息

**字段说明**：
```json
[
  {
    "file_path": "E:\\MyGame\\GameVideos\\lolhighlight\\...\\record\\record_abc123...mp4",
    "file_name": "record_abc123...",
    "hero_name": "梅尔",  // 中文名
    "hero_type": "法师",      // 英雄类型：法师/战士/刺客/坦克/辅助
    "battle_result": "win",     // 对局结果：win/fail
    "KDA": "3/5/7",       // KDA数据：击杀/死亡/助攻
    "label_status": "pending"  // 标注状态：pending/labeling/completed/skipped
  }
]
```

**标注状态说明**：
- `pending` - 待标注
- `labeling` - 标注进行中
- `completed` - 标注已完成
- `skipped` - 已跳过（如低KDA败局等低质量数据）

---

## 文件关系说明

### data/hero_states/video_status.json

- **管理对象**：标注工具管理的所有视频
- **命名方式**：视频文件名作为唯一标识
- **英雄名称**：使用中文翻译名（如：塞拉斯、蒙多、瑞兹、蔚）
- **一致性**：英雄名称在标注工具内使用翻译名

---

### E:\MyGame\GameVideos\record_details.json

- **管理对象**：游戏脚本自动生成的对局信息
- **命名方式**：文件名+时间戳
- **英雄名称**：使用中文名（如：梅尔、娜美）
- **数据来源**：游戏完成后自动记录

---

## 使用游戏资料标注

### 查询游戏资料

在标注前，可以先查询 `record_details.json` 了解视频信息：

```python
import json

# 读取游戏资料
with open('E:/MyGame/GameVideos/record_details.json', 'r', encoding='utf-8') as f:
    records = json.load(f)

# 查找特定视频
for record in records:
    file_path = record['file_path']
    file_name = record['file_name']
    hero_name = record['hero_name']
    hero_type = record['hero_type']
    battle_result = record['battle_result']
    kda = record['KDA']
    label_status = record.get('label_status', 'pending')

    print(f"视频: {file_name}")
    print(f"英雄: {hero_name} ({hero_type})")
    print(f"结果: {battle_result}")
    print(f"标注状态: {label_status}")
    if kda:
        k, d, a = kda.split('/')
        print(f"KDA: {k}/{d}/{a} ({int(k)+int(d)+int(a)})")
```

### 批量标注建议

#### 根据对局信息确定标注优先级

| 对局质量 | KDA | 优先级 | 标注策略 |
|---------|-----|--------|----------|
| 胜局 + 高KDA ≥ 1.0 | ⭐⭐⭐⭐ | 完整标注，价值最高 |
| 败局 + 高KDA ≥ 1.0 | ⭐⭐⭐⭐ | 完整标注，价值高 |
| 败局 + 中KDA (0.5-1.0) | ⭐⭐⭐ | 完整标注，价值中 |
| 败局 + 低KDA < 0.5 | ⭐⭐⭐ | 粡查，或快速浏览 |
| 胜局 + 低KDA < 0.5 | ⭐⭐⭐⭐ | 没查，快速浏览 |

#### 批局价值分析

- ✅ **高KDA败局（KDA ≥ 1.0）**：有输出、有参与，学习价值高
- ✅ **中KDA败局（0.5 ≤ KDA < 1.0）**：有一定输出，正常学习
- ❌ **低KDA败局（KDA < 0.5）**：梦游局，无输出，无学习价值

#### 胜局标注规则

- **高KDA败局**：完整标注，学习"正确"状态的重要性
- **中KDA败局**：可跳过或快速标注
- **低KDA败局**：建议跳过，只保留高质量数据

#### 使用label_status过滤

根据 `label_status` 字段过滤需要标注的视频：

```python
import json

# 读取游戏资料
with open('E:/MyGame/GameVideos/record_details.json', 'r', encoding='utf-8') as f:
    records = json.load(f)

# 只标注未跳过的视频
pending_records = [r for r in records if r.get('label_status', 'pending') != 'skipped']

for record in pending_records:
    print(f"待标注: {record['file_name']} ({record['hero_name']})")
```

---

## 文件命名与英雄名称

### video_status.json（标注工具）

**英雄名称使用翻译名：**
| 中文名 | 英文名 | 文件示例 |
|--------|--------|----------|
| 塞拉斯 | Sylas | record20260123-085500-瑞兹-fail-KDA-21-15-16.mp4 |
| 蒙多 | Morgana | record20260123-215641-蒙多-win.mp4 |
| 蔚 | Vayne | record20260123-223655-蔚-win.mp4 |
| 瑞兹 | Ryze | record20260123-085500-瑞兹-fail-KDA-21-15-16.mp4 |

### record_details.json（游戏资料）

**中文名用于数据目录，英文名用于文件名：**
| 中文名 | 英文名 | 英雄类型 | 说明 |
|--------|--------|----------|-----------|
| 梅尔 | Mel | 法师 | 控制型英雄 |
| 娜美 | Nami | 辅助 | 腰利辅助，保护队友 |
| 蔚 | Vayne | 射手 | 远程消耗，收割后期 |
| 瑞兹 | Ryze | 刺客 | 爆机突进，高爆发 |
| 塞拉斯 | Sylas | 法师 | 夺制型英雄 |

---

## 标注与训练的配合

### 当前数据状态（截至 2026-01-25）

**已标注英雄（4个）：**
- 塞拉斯（胜局，290帧）
- 蒙多（胜局，300帧）
- 蔚（胜局，289帧）
- 瑞兹（胜局，274帧）

**总计：1153帧**

**待标注英雄（根据 record_details.json）：**
- 梅尔（胜局，预计300帧）
- 娜美（败局，预计300帧）

---

## 策略总结

### 标注工具（video_status.json）

- **唯一标识**：视频文件名（如：record20260123-222005-塞拉斯-win.mp4）
- **英雄名称**：中文名（如：塞拉斯）
- **状态跟踪**：pending/labeling/completed
- **防重复标注**：尝试标注已完成视频时会提示
- **训练状态**：`trained`字段标识是否已用于训练
- **训练信息**：`training_info`记录模型路径、训练日期、验证准确率等

### 游戏资料（record_details.json）

- **信息来源**：游戏脚本自动生成
- **自动记录**：每局完成后记录
- **数据字段**：文件名、英雄类型、对局结果、KDA等
- **批量标注参考**：根据对局质量确定优先级

---

## 下一步

### 用现有1153帧数据开始训练

1. **数据准备**：
   - 检查4个英雄的数据分布
   - 创建数据加载器
   - 数据划分（训练/验证/测试）

2. **模型训练**：
   - 训练视觉编码器
   - 训练状态分类器
   - 验证模型性能

3. **推理测试**：
   - 加载训练好的模型
   - 实时推理脚本测试
   - 评估预测准确性

4. **性能优化**：
   - 如果效果不佳，继续收集更多数据
   - 如果效果良好，准备在线学习

你现在想开始训练吗？

输出示例：
```
================================================================================
视频标注状态 (当前策略版本: v1.0)
================================================================================
✅ game1.webm (Lux)
   状态: completed | 进度: 200/200 (100%)
   完成时间: 2026-01-23T16:00:00

🏷️️ game2.webm (Lux)
   状态: labeling | 进度: 50/200 (25%)

⏸️ game3.webm (Lux)
   状态: pending | 进度: 0/0 (0%)
```

### 状态说明

- **pending** ⏸️：视频已注册但尚未开始标注
- **labeling** 🏷️️：标注进行中
- **completed** ✅：标注已完成（所有帧都已标注）

### 防止重复标注

当你尝试标注一个已完成的视频时，工具会提示：

```
⚠️ game1.webm 已标注完成！
完成时间: 2026-01-23T16:00:00
标注进度: 200/200

是否继续重新标注? (y/N):
```

输入`y`可重新标注，其他选项或直接回车将取消。

### 状态文件位置

视频状态保存在：
- `data/hero_states/video_status.json`

文件格式：
```json
{
  "abc123...": {
    "video_path": "/path/to/game1.webm",
    "hero_name": "Lux",
    "status": "completed",
    "total_frames": 200,
    "labeled_frames": 200,
    "created_at": "2026-01-23T15:00:00",
    "completed_at": "2026-01-23T16:00:00"
  }
}
```

### 游戏资料文件

**文件位置**: `E:\MyGame\GameVideos\record_details.json`

**作用**: 保存游戏对局的详细资料，方便标注时获取视频信息

**文件格式**:
```json
[
  {
    "file_path": "E:\\MyGame\\GameVideos\\lolhighlight\\...\\record\\record20260124-161447.mp4",
    "file_name": "record20260124-161447.mp4",
    "hero_name": "梅尔",
    "hero_type": "法师",
    "battle_result": "win",
    "KDA": ""
  },
  {
    "file_path": "E:\\MyGame\\GameVideos\\lolhighlight\\...\\record\\record20260124-163753.mp4",
    "file_name": "record20260124-163753.mp4",
    "hero_name": "娜美",
    "hero_type": "辅助",
    "battle_result": "fail",
    "KDA": "1/17/18"
  }
]
```

**字段说明**:
- `file_path`: 视频文件的完整路径
- `file_name`: 视频文件名
- `hero_name`: 英雄名称（中文翻译名）
- `hero_type`: 英雄类型（法师、战士、刺客、辅助、坦克）
- `battle_result`: 对局结果（win/fail）
- `KDA`: KDA数据（格式：击杀/死亡/助攻）

### 使用游戏资料

#### 标注前检查

在标注前，可以读取 `record_details.json` 检查视频信息：

```python
import json

# 读取游戏资料
with open('E:/MyGame/GameVideos/record_details.json', 'r', encoding='utf-8') as f:
    records = json.load(f)

# 查找特定视频
for record in records:
    if 'game1.webm' in record['file_name']:
        hero = record['hero_name']
        result = record['battle_result']
        kda = record['KDA']
        print(f"英雄: {hero}, 结果: {result}, KDA: {kda}")
```

#### 批量标注建议

根据 `record_details.json` 中的信息，可以批量标注视频：

```bash
# 方法1：遍历 record_details.json，批量标注所有视频
python scripts/label_tool.py --hero <英雄名称> --video <视频路径>

# 方法2：根据对局结果优先标注
# 优先标注胜局
# 高KDA败局次之
# 低KDA败局最后或跳过
```

### 文件对比

| 文件 | 作用 | 更新方式 |
|------|------|----------|
| `record_details.json` | 游戏对局资料 | 游戏完成后自动更新 |
| `video_status.json` | 视频标注状态 | 标注工具运行时更新 |

**注意事项**：
- `record_details.json` 由游戏脚本自动更新，无需手动编辑
- `video_status.json` 记录标注进度，工具自动管理
- 英雄名称以 `record_details.json` 为准（中文名）
- 数据目录使用中文名作为目录名

输出示例：
```
================================================================================
视频标注状态
================================================================================
✅ game1.webm (Lux)
   状态: completed | 进度: 200/200 (100%)
   完成时间: 2026-01-23T16:00:00

🏷️ game2.webm (Lux)
   状态: labeling | 进度: 50/200 (25%)

⏸️ game3.webm (Lux)
   状态: pending | 进度: 0/0 (0%)
```

### 状态说明

- **pending** ⏸️：视频已注册但尚未开始标注
- **labeling** 🏷️：标注进行中
- **completed** ✅：标注已完成（所有帧都已标注）

### 防止重复标注

当你尝试标注一个已完成的视频时，工具会提示：

```
⚠️ game1.webm 已标注完成！
完成时间: 2026-01-23T16:00:00
标注进度: 200/200

是否继续重新标注? (y/N):
```

输入`y`可重新标注，其他选项或直接回车将取消。

### 状态文件位置

视频状态保存在：
- `data/hero_states/video_status.json`

文件格式：
```json
{
  "abc123...": {
    "video_path": "/path/to/game1.webm",
    "hero_name": "Lux",
    "status": "completed",
    "total_frames": 200,
    "labeled_frames": 200,
    "created_at": "2026-01-23T15:00:00",
    "completed_at": "2026-01-23T16:00:00"
  }
}
```

### 支持的视频格式

除了之前的 `.mp4`, `.avi`, `.mkv`，现在还支持 `.webm` 格式。

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

### 4. 大乱斗模式特殊场景标注

**大乱斗商店规则：**
- ❌ 没有传统"回城"机制
- ✅ 只有泉水区域，任何时间都能买东西
- ✅ 按P快捷键打开商店买东西
- ✅ 死亡后立即复活，可在商店买东西

**特殊场景标注规则：**

| 场景 | 标注 | 理由 |
|------|------|------|
| 在泉水买装备 | **6-买装备** | 买装备状态 |
| 死亡后在商店买东西 | **6-买装备** | 买装备状态 |
| 开局出发时在商店买东西 | **6-买装备** | 买装备状态 |
| 按P键打开商店买东西 | **6-买装备** | 买装备状态 |
| 买完装备后短暂静止 | **6-买装备** | 买装备状态 |
| 离开泉水进入战场 | **1-移动** | 准备战斗 |
| 在商店里同时移动 | **1-移动** | 移动为主要动作 |

**注意：**
- 大乱斗没有"回城"按钮，商店一直在泉水
- 买装备是重要决策时刻，需要AI学习
- 新增"买装备"标签用于区分真正的死亡和买装备状态

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

## 标注策略v2.0（新增买装备标签）

### 更新内容

**从v1.0升级到v2.0的主要变化：**
- 新增第6个标签："买装备"
- 更新策略版本号：v1.0 → v2.0
- 添加快捷键6用于标注买装备状态

### 为什么要添加"买装备"标签？

**大乱斗模式特殊性：**
- 死亡后立即复活，可以在商店买装备
- 买装备是重要的决策时刻，需要AI学习
- 买装备和死亡是两种完全不同的状态

**v1.0的问题：**
- 买装备场景被标注为"死亡"（或"等待"）
- AI无法区分真正的死亡和买装备
- 导致推理时可能做出错误操作

**v2.0的改进：**
- 明确区分"死亡"和"买装备"
- 视觉编码器能学习到"买装备"的独特特征
- DQN能学会"买装备"状态应该选择"等待"动作

### 标签对比

| 标签 | v1.0 | v2.0 |
|------|------|------|
| 移动 | ✅ | ✅ |
| 攻击 | ✅ | ✅ |
| 技能 | ✅ | ✅ |
| 受伤 | ✅ | ✅ |
| 死亡 | ✅ | ✅ |
| 买装备 | ❌ | ✅ **新增** |

### 旧数据处理

**已有数据的兼容性：**
- ✅ 旧数据（v1.0标注）可以正常加载
- ✅ 5个标签的数据不会丢失
- ⚠️ 建议：重新标注旧数据中"买装备"的帧

**重新标注建议：**
1. 浏览所有"死亡"标签的帧
2. 识别买装备场景（商店UI、泉水环境）
3. 将买装备的帧标注为"6-买装备"
4. 保存更新后的数据

### 训练影响

**阶段1：视觉编码器训练**
- 训练数据：6类别（包含"买装备"）
- 模型输出：6个类别的概率
- 预期准确率：75-85%

**阶段2：DQN训练**
- 输入：256维特征向量（包含"买装备"特征）
- 输出：8个动作
- 间接效果：DQN学会"买装备"→"等待"的映射

**阶段3：推理**
- 买装备场景被正确识别
- DQN选择"等待"动作（动作7）
- 避免错误操作（如移动、攻击）

---

**文档版本**: 2.0
**创建日期**: 2026-01-21
**最后更新**: 2026-01-26
**更新内容**: 新增"买装备"标签，更新到v2.0

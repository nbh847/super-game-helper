# 架构设计

## 目录结构

```
games/lol-helper/
├── backend/
│   ├── core/
│   │   ├── game_state.py       # 游戏状态识别
│   │   ├── action_executor.py  # 操作执行
│   │   └── ai_engine.py        # AI决策
│   │
│   ├── ai/
│   │   ├── pretrain/
│   │   │   ├── replay_parser.py    # .rofl 录像解析
│   │   │   ├── state_extractor.py # 提取游戏状态
│   │   │   ├── action_extractor.py # 提取玩家操作
│   │   │   ├── data_loader.py     # 训练数据加载
│   │   │   ├── model.py           # 策略网络
│   │   │   └── trainer.py         # 训练器
│   │   ├── finetune/
│   │   │   ├── arena_env.py        # 大乱斗环境
│   │   │   ├── agent.py
│   │   │   └── trainer.py
│   │   └── models/
│   │
│   ├── utils/
│   │   ├── screen_capture.py   # 高效屏幕截取
│   │   ├── image_recognition.py # OCR识别+目标检测
│   │   ├── input_simulator.py  # 鼠标/键盘模拟
│   │   └── human_behavior.py   # 人类行为模拟
│   │
│   ├── data/
│   │   ├── replay_converter.py # 转换 .rofl 到训练数据
│   │   ├── dataset/
│   │   │   └── raw/            # 原始录像
│   │   └── processed/          # 处理后的训练数据
│   │
│   ├── config/
│   │   ├── settings.py
│   │   ├── hero_profiles/
│   │   │   ├── tank/
│   │   │   ├── mage/
│   │   │   ├── marksman/
│   │   │   ├── assassin/
│   │   │   └── support/
│   │   └── supported_heroes.json # 已支持的英雄列表
│   │
│   ├── main.py
│   └── requirements.txt
│
├── docs/               # 文档
│   ├── README.md
│   ├── design_proposal.md
│   ├── architecture.md      # 本文件
│   ├── anti_detection.md
│   └── api_reference.md
│
└── README.md
```

## 核心模块说明

### 1. 游戏状态识别 (game_state.py)

**功能**
- 屏幕区域识别：小地图、英雄血量、技能冷却、金币
- 英雄位置和朝向
- 敌方英雄位置和威胁度
- 己方/敌方小兵位置和血量
- 防御塔位置和攻击范围

### 2. 操作执行器 (action_executor.py)

**功能**
- 基础操作：移动、普攻、技能释放、回血
- 鼠标/键盘操作随机化
- 参考现有 lushi-cheater 的人类行为模拟

### 3. AI 决策引擎 (ai_engine.py)

**功能**
- 根据游戏状态生成操作决策
- 支持行为克隆和强化学习两种模式
- 针对不同英雄类型调整策略

### 4. 录像解析 (replay_parser.py)

**功能**
- 解析 .rofl 文件结构
- 提取关键帧（每秒30帧）
- 获取玩家操作序列（移动、技能、普攻）

**依赖**
- lol-replay-parser 库

### 5. 数据提取 (state_extractor.py & action_extractor.py)

**状态提取**
- 从每帧提取状态特征
- 归一化和特征工程

**动作提取**
- 提取对应的玩家操作
- 动作编码（移动方向、技能ID、目标等）

### 6. 数据加载器 (data_loader.py)

**功能**
- 加载已处理的训练数据
- 数据增强（镜像、平移、时间扭曲）
- 批量加载和预处理

### 7. 策略模型 (model.py)

**模型架构**
- 卷积神经网络（处理视觉输入）
- LSTM/GRU（处理序列信息）
- 全连接层（输出动作概率）

### 8. 训练器 (trainer.py)

**预训练**
- 使用行为克隆从数据学习
- 监督学习损失函数（交叉熵）

**微调**
- 使用强化学习（PPO/DQN）
- 自我对弈或与环境交互

### 9. 大乱斗环境 (arena_env.py)

**功能**
- 模拟极地大乱斗游戏环境
- 定义状态空间和动作空间
- 奖励函数设计

### 10. 强化学习智能体 (agent.py)

**算法选择**
- PPO（Proximal Policy Optimization）
- DQN（Deep Q-Network）

### 11. 屏幕截取 (screen_capture.py)

**功能**
- 高效截取游戏窗口
- 截取特定区域（小地图、技能栏等）
- 优化性能（缓存、区域更新）

### 12. 图像识别 (image_recognition.py)

**功能**
- OCR 识别血量、金币数值
- 目标检测（YOLO）：英雄、小兵、防御塔
- 位置识别和追踪

**依赖**
- OpenCV
- PaddleOCR
- YOLO

### 13. 输入模拟 (input_simulator.py)

**功能**
- 鼠标移动和点击
- 键盘按键模拟
- 支持快捷键组合

**依赖**
- PyAutoGUI

### 14. 人类行为模拟 (human_behavior.py)

**防检测机制**
- APM 控制：150-250（玩家正常范围）
- 操作延迟随机：200-600ms
- 鼠标轨迹随机性（贝塞尔曲线）
- 偶尔"失误"（空技能、走位失误）
- 定期"休息"（暂停操作 1-3 秒）

### 15. 录像转换器 (replay_converter.py)

**功能**
- 批量转换 .rofl 文件到训练数据
- 过滤和清洗数据
- 存储为标准格式

## 英雄分类与配置

### 英雄类型
- **坦克**：高生命值，前排抗伤，控制技能
- **法师**：远程输出，范围技能，高爆发
- **射手**：持续输出，普攻为主，需要保护
- **刺客**：高机动性，切入后排，单点爆发
- **辅助**：保护队友，增益技能，视野控制

### 英雄配置文件结构
```json
{
  "hero_name": "Lux",
  "type": "mage",
  "skills": {
    "q": "light_binding",
    "w": "prismatic_barrier",
    "e": "lucent_singularity",
    "r": "final_spark"
  },
  "playstyle": {
    "preferred_range": 550,
    "aggressiveness": 0.7,
    "team_fight_role": "damage"
  }
}
```

## 系统架构图

### 训练流程

```
.rofl 录像
    ↓
录像解析
    ↓
状态提取 → 动作提取
    ↓         ↓
  训练数据 ←——
    ↓
数据增强
    ↓
预训练（行为克隆）
    ↓
基础模型
    ↓
微调（强化学习）
    ↓
最终模型
```

### 实时运行流程

```
游戏窗口
    ↓
屏幕截取
    ↓
图像识别
    ↓
游戏状态
    ↓
AI决策
    ↓
人类行为模拟
    ↓
操作执行
    ↓
游戏客户端
```

## 数据流

### 训练数据流
```
原始录像 (.rofl)
  → 解析 → 关键帧
  → 特征提取 → (状态, 动作) 对
  → 数据增强 → 训练数据集
  → 模型训练 → 策略模型
```

### 运行时数据流
```
屏幕图像 (1920x1080)
  → 区域截取 → (小地图、技能栏、游戏画面)
  → 图像识别 → 游戏状态向量
  → AI推理 → 动作指令
  → 行为模拟 → 键鼠操作
  → 游戏响应 → 新状态
```

## 配置管理

### settings.py
```python
class Settings:
    # 模型配置
    MODEL_PATH = "models/pretrained_model.pth"
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001

    # 游戏配置
    GAME_WINDOW_NAME = "League of Legends"
    SCREEN_WIDTH = 1920
    SCREEN_HEIGHT = 1080

    # AI配置
    HERO_TYPE = "mage"
    AGGRESSIVENESS = 0.7

    # 防检测配置
    TARGET_WIN_RATE = 0.55
    MIN_APM = 150
    MAX_APM = 250
```

## 性能优化

### 屏幕截取优化
- 只截取需要的区域
- 使用缓存避免重复截取
- 降低分辨率（不影响识别）

### 模型推理优化
- 使用 GPU 加速
- 批量推理
- 模型量化（可选）

### 内存优化
- 及时释放不需要的数据
- 使用生成器处理大数据
- 分批加载

## 错误处理

### 异常处理策略
- 游戏窗口未找到：等待重试
- 图像识别失败：使用上一帧结果
- 操作执行失败：跳过，记录日志
- 模型推理失败：使用默认动作

### 日志记录
- 操作日志：记录所有执行的操作
- 性能日志：记录 FPS、延迟等
- 错误日志：记录异常和错误
- 统计日志：记录胜率、KDA等

## 相关文档

- [设计提案](design_proposal.md) - 整体设计和实施计划
- [防检测策略](anti_detection.md) - 完整的防检测实施方案
- [模块API参考](api_reference.md) - 代码接口和实现细节

---

**文档版本**: 1.0
**最后更新**: 2026-01-19

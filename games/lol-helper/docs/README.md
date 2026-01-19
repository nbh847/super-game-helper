# LOL 极地大乱斗 AI 助手

## 项目简介

本项目旨在开发一个英雄联盟极地大乱斗模式的 AI 助手，通过行为克隆和强化学习技术，训练能够自主操作的 AI 玩家。项目强调模拟人类操作行为，避免被游戏官方检测为外挂。

## 核心目标

- 全自动操作极地大乱斗模式
- 通过 AI 学习和强化游戏能力
- 模拟人类操作，降低检测风险
- 支持全英雄分类（坦克/法师/射手/刺客/辅助）
- 优先战斗策略，兼顾补刀

## 技术路线

- 预训练（行为克隆）+ 微调（强化学习）
- 渐进式技能复杂度提升
- 基于现有 .rofl 游戏录像数据训练

## 快速开始

### 环境要求

- Python 3.9+
- CUDA 11.8+（GPU 加速，可选）
- 推荐配置：RTX5060 8GB 显卡 + 16GB 内存

### 安装依赖

```bash
cd games/lol-helper/backend
pip install -r requirements.txt
```

### 使用说明

详细的实施步骤和模块说明请参考以下文档：

- [设计提案](design_proposal.md) - 整体设计和实施计划
- [架构设计](architecture.md) - 目录结构和模块说明
- [防检测策略](anti_detection.md) - 防检测详细策略
- [模块API参考](api_reference.md) - 代码接口和实现细节

## 渐进式版本

### V1：基础操作
- 基础走位
- 补刀（小兵）
- 识别安全距离

### V2：+ 1 个主要技能
- 识别技能命中范围
- 合理时机释放
- 结合普攻输出

### V3：+ 2-3 个技能组合
- 技能连招
- 技能搭配使用
- 冷却时间管理

### V4：完整技能 + 召唤师技能
- 复杂连招
- 召唤师技能使用
- 全技能配合

## 英雄分类

- **坦克**：高生命值，前排抗伤，控制技能
- **法师**：远程输出，范围技能，高爆发
- **射手**：持续输出，普攻为主，需要保护
- **刺客**：高机动性，切入后排，单点爆发
- **辅助**：保护队友，增益技能，视野控制

## 技术栈

### Python 依赖

```txt
# 录像解析
lol-replay-parser

# 深度学习
torch
torchvision
stable-baselines3
gymnasium

# 图像处理
opencv-python
paddleocr
ultralytics

# 操作模拟
pyautogui
pynput

# 数据处理
numpy
pandas
h5py

# 工具
pillow
pyyaml
tqdm
```

## 数据现状

### 已有资源

- 格式：.rofl 游戏录像
- 数量：中等（10-100 局）
- 英雄覆盖：少数英雄

### 数据处理流程

1. 将 .rofl 文件放入 `data/dataset/raw/`
2. 运行 `replay_converter.py` 批量转换
3. 处理后的数据存储到 `data/dataset/processed/`
4. 训练时使用 `data_loader.py` 加载

## 目录结构

```
games/lol-helper/
├── backend/
│   ├── core/           # 核心业务逻辑
│   ├── ai/             # AI训练模块
│   ├── utils/          # 工具模块
│   ├── data/           # 数据模块
│   ├── config/         # 配置文件
│   ├── main.py         # 主入口
│   └── requirements.txt
├── docs/               # 文档
│   ├── README.md       # 本文件
│   ├── design_proposal.md
│   ├── architecture.md
│   ├── anti_detection.md
│   └── api_reference.md
└── README.md
```

## 参考项目

- [lushi-cheater](https://github.com/nbh847/lushi_cheater) - 人类行为模拟
- [League of Legends Replay Parser](https://github.com/Paxti/league_of_legends_replay_parser) - 录像解析
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) - 强化学习框架
- [YOLOv8](https://github.com/ultralytics/ultralytics) - 目标检测

## 许可证

MIT License

---

**文档版本**: 1.0
**最后更新**: 2026-01-19

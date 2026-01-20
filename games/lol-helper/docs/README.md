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

### 技术路线验证

**已验证的成功案例**：

1. **马斯克 Grok AI**（2026年1月）
   - ✅ 92%胜率登顶韩服（56局52胜）
   - ✅ 精通22个英雄
   - ✅ 验证了行为克隆+强化学习组合的可行性
   - ⚠️ 但过于规律的作息和高胜率引起关注（需避免）

2. **DeepMind AlphaStar**（2019年）
   - ✅ 首个达到宗师级的RTS游戏AI
   - ✅ 使用多智能体强化学习
   - ✅ Actor-Critic + LSTM + PPO 组合
   - ✅ 通过自我对弈提升

3. **TLoL 开源项目**
   - ✅ 833场挑战者金克斯对局数据集
   - ✅ 完整的录像解析和数据处理流程
   - ✅ 基于深度学习的游戏AI框架

4. **LeagueAI 图像识别框架**
   - ✅ OpenCV + PyTorch 的图像识别方案
   - ✅ 实时游戏状态识别
   - ✅ 目标检测和OCR识别集成

5. **腾讯王者荣耀 AI**（AAAI 2020）
   - ✅ 深度强化学习框架
   - ✅ 能击败顶尖职业选手
   - ✅ 类似的MOBA游戏环境

## 快速开始

### 环境要求

- Python 3.9+
- CUDA 11.8+（GPU 加速，可选）
- 推荐配置：RTX5060 8GB 显卡 + 16GB 内存
- 个人用户配置：适合个人开发使用

### 个人用户简化方案

**适用场景**：
- 只玩大乱斗模式（ARAM）
- 支持10-15个常用英雄
- V1版本：基础操作（移动+普攻+补刀）
- 预期胜率：人机80%+，正常玩家30-50%

**资源占用**（针对RTX5060 8GB优化）：
- 训练显存：2-3GB
- 推理显存：1.5-2.5GB
- 内存占用：4-6GB
- 开发时间：11-18天

**简化策略**：
- 小模型架构（hidden_size=128）
- 降低batch_size=16
- 每英雄只需10-20局录像
- 使用YOLOv8n（nano版本）
- 跳帧处理（每4帧处理一次）

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

### 商业案例
- **马斯克 Grok AI** - 92%胜率登顶韩服（2026年1月）
- **DeepMind AlphaStar** - 星际争霸II宗师级AI

### 开源项目
- [TLoL](https://github.com/MiscellaneousStuff/tlol) - 英雄联盟深度学习AI
- [LeagueAI](https://github.com/Oleffa/LeagueAI) - 图像识别框架
- [TLoL Scraper](https://github.com/MiscellaneousStuff/tlol-scraper) - 录像提取器

### 数据集
- [HuggingFace: league-of-legends-decoded-replay-packets](https://huggingface.co/datasets/maknee/league-of-legends-decoded-replay-packets) - 1TB+录像数据（700k+）
- [TLoL Dataset](https://github.com/MiscellaneousStuff/tlol) - 833场挑战者对局

### 技术库
- [League of Legends Replay Parser](https://github.com/Paxti/league_of_legends_replay_parser) - 录像解析
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) - 强化学习框架
- [YOLOv8](https://github.com/ultralytics/ultralytics) - 目标检测
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - OCR识别

## 许可证

MIT License

---

**文档版本**: 1.1
**最后更新**: 2026-01-20
**更新内容**：添加Grok AI调研结果、个人用户简化方案、开源项目参考

# LOL 极地大乱斗 AI 助手设计提案

## 项目概述

本项目旨在开发一个英雄联盟极地大乱斗模式的 AI 助手，通过行为克隆和强化学习技术，训练能够自主操作的 AI 玩家。项目强调模拟人类操作行为，避免被游戏官方检测为外挂。

### 核心目标
- 全自动操作极地大乱斗模式
- 通过 AI 学习和强化游戏能力
- 模拟人类操作，降低检测风险
- 支持全英雄分类（坦克/法师/射手/刺客/辅助）
- 优先战斗策略，兼顾补刀

### 技术路线
- 预训练（行为克隆）+ 微调（强化学习）
- 渐进式技能复杂度提升
- 基于现有 .rofl 游戏录像数据训练

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
├── docs/
│   └── design_proposal.md       # 本文档
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

**关键接口**
```python
class GameState:
    def __init__(self):
        self.hero_position = None
        self.hero_health = None
        self.hero_mana = None
        self.skills_cooldown = []
        self.enemy_positions = []
        self.minion_positions = []
        self.tower_position = None
        self.gold = None

    def update_from_screen(self, screenshot):
        """从屏幕截图更新游戏状态"""
        pass

    def to_tensor(self):
        """转换为神经网络输入张量"""
        pass
```

### 2. 操作执行器 (action_executor.py)

**功能**
- 基础操作：移动、普攻、技能释放、回血
- 鼠标/键盘操作随机化
- 参考现有 lushi-cheater 的人类行为模拟

**关键接口**
```python
class ActionExecutor:
    def move_to(self, target_pos):
        """移动到目标位置（模拟人类速度）"""
        pass

    def attack_target(self, target_pos):
        """攻击目标"""
        pass

    def cast_skill(self, skill_key, target_pos=None):
        """释放技能"""
        pass

    def use_heal(self):
        """使用治疗道具"""
        pass
```

### 3. AI 决策引擎 (ai_engine.py)

**功能**
- 根据游戏状态生成操作决策
- 支持行为克隆和强化学习两种模式
- 针对不同英雄类型调整策略

**关键接口**
```python
class AIEngine:
    def __init__(self, model_path, hero_type):
        self.model = self.load_model(model_path)
        self.hero_type = hero_type

    def decide_action(self, game_state):
        """根据游戏状态决策下一步操作"""
        action = self.model.predict(game_state.to_tensor())
        return action

    def update_policy(self, reward):
        """根据奖励更新策略（强化学习模式）"""
        pass
```

### 4. 录像解析 (replay_parser.py)

**功能**
- 解析 .rofl 文件结构
- 提取关键帧（每秒30帧）
- 获取玩家操作序列（移动、技能、普攻）

**依赖**
- lol-replay-parser 库

**关键接口**
```python
class ReplayParser:
    def parse_replay(self, replay_path):
        """解析 .rofl 文件"""
        pass

    def extract_frames(self, fps=30):
        """提取关键帧"""
        pass

    def extract_actions(self):
        """提取玩家操作序列"""
        pass
```

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

**关键接口**
```python
class DataLoader:
    def __init__(self, data_path, batch_size=32):
        pass

    def get_batch(self):
        """获取一个批次的训练数据"""
        pass

    def augment_data(self, data):
        """数据增强"""
        pass
```

### 7. 策略模型 (model.py)

**模型架构**
- 卷积神经网络（处理视觉输入）
- LSTM/GRU（处理序列信息）
- 全连接层（输出动作概率）

**关键接口**
```python
class PolicyModel(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        # CNN + LSTM + FC
        pass

    def forward(self, state):
        """前向传播"""
        pass

    def predict(self, state):
        """预测动作"""
        pass
```

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

**关键接口**
```python
class ArenaEnv(gym.Env):
    def __init__(self):
        self.observation_space = ...
        self.action_space = ...

    def step(self, action):
        """执行动作，返回状态、奖励、是否结束"""
        pass

    def reset(self):
        """重置环境"""
        pass
```

### 10. 强化学习智能体 (agent.py)

**算法选择**
- PPO（Proximal Policy Optimization）
- DQN（Deep Q-Network）

**关键接口**
```python
class RLA agent:
    def __init__(self, env):
        self.env = env
        self.model = ...

    def train(self, num_episodes):
        """训练智能体"""
        pass

    def act(self, state):
        """根据状态选择动作"""
        pass
```

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

**关键接口**
```python
class HumanBehavior:
    def add_random_delay(self, min_ms=200, max_ms=600):
        """添加随机延迟"""
        pass

    def simulate_mouse_movement(self, start, end):
        """模拟人类鼠标轨迹（贝塞尔曲线）"""
        pass

    def random_mistake(self, probability=0.05):
        """以一定概率模拟失误操作"""
        pass

    def occasional_rest(self, min_sec=1, max_sec=3):
        """偶尔休息"""
        pass
```

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

### 针对少数英雄的策略扩展

**阶段 1：基础模型**
- 使用现有英雄数据训练基础模型
- 支持已有英雄的完整策略

**阶段 2：数据增强**
- 镜像翻转（左右对称）
- 位置平移（小范围）
- 时间扭曲（速度调整）

**阶段 3：迁移学习**
- 将基础模型迁移到新英雄
- 共享底层特征提取器
- 冻结部分网络层

**阶段 4：少量新英雄数据微调**
- 收集少量新英雄对局数据
- 微调最后几层网络
- 快速适配新英雄

## 渐进式技能复杂度

### V1：基础操作（移动 + 普攻）
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

## 战斗优先策略

### 优先级顺序
1. **生存**：血量低时优先撤退/回血
2. **团战**：敌方英雄集结时优先战斗
3. **击杀机会**：敌方残血时优先追击
4. **输出**：在安全位置输出伤害
5. **补刀**：空闲时补刀积累经济

### 威胁度评估
- 敌方英雄血量
- 敌方英雄技能冷却
- 敌方英雄与己方距离
- 防御塔覆盖范围

### 团战策略
- **坦克**：前排承伤，控制敌方
- **法师**：后排输出，范围技能
- **射手**：安全位置，持续输出
- **刺客**：切入后排，击杀脆皮
- **辅助**：保护队友，提供增益

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
ultralytics  # YOLOv8

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

### 系统依赖
- Python 3.9+
- CUDA 11.8+（GPU 加速，可选）

## 实施计划

### 阶段 1：基础框架搭建
- [ ] 创建目录结构
- [ ] 配置文件模板
- [ ] 基础模块骨架
- [ ] 依赖库安装和测试

### 阶段 2：数据处理
- [ ] 录像解析模块
- [ ] 状态/动作提取
- [ ] 数据加载器
- [ ] 数据增强测试

### 阶段 3：模型训练（V1）
- [ ] 策略网络模型
- [ ] 预训练器
- [ ] 基础操作训练
- [ ] 模型评估

### 阶段 4：实时游戏识别
- [ ] 屏幕截取优化
- [ ] 图像识别模块
- [ ] 游戏状态识别
- [ ] 实时性能优化

### 阶段 5：操作执行器
- [ ] 鼠标/键盘模拟
- [ ] 人类行为模拟
- [ ] 防检测机制
- [ ] 操作准确性测试

### 阶段 6：集成与测试
- [ ] 端到端集成
- [ ] 实战测试
- [ ] 性能优化
- [ ] 错误处理和日志

### 阶段 7：扩展（V2+）
- [ ] 技能释放模块
- [ ] 更复杂策略
- [ ] 强化学习微调
- [ ] 新英雄支持

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

## 风险与挑战

### 技术挑战
1. **实时性能**：屏幕识别和决策需要快节奏响应
2. **状态不完整**：屏幕截图信息有限，需要推理
3. **数据不足**：少数英雄数据需要数据增强和迁移学习
4. **泛化能力**：不同场景和对手的适应

### 风险控制
1. **分阶段验证**：每个阶段充分测试再推进
2. **模块化设计**：各模块独立开发和测试
3. **渐进式复杂度**：从简单到复杂逐步提升
4. **持续监控**：运行时记录日志和性能指标

### 防检测措施
- 严格的人类行为模拟
- APM 和操作频率控制
- 定期更新策略（避免固定模式）
- 监控游戏客户端的反外挂机制

## 后续优化方向

1. **多模态输入**：结合游戏 API 数据（如果可用）
2. **对手建模**：学习和适应不同对手风格
3. **团队协作**：多 AI 配合（模拟团队对局）
4. **在线学习**：实战中持续优化策略
5. **可视化分析**：训练和操作过程可视化
6. **配置化管理**：支持灵活调整策略参数

## 参考项目

- [lushi-cheater](https://github.com/nbh847/lushi_cheater) - 人类行为模拟
- [League of Legends Replay Parser](https://github.com/Paxti/league_of_legends_replay_parser) - 录像解析
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) - 强化学习框架
- [YOLOv8](https://github.com/ultralytics/ultralytics) - 目标检测

---

**文档版本**: 1.0
**最后更新**: 2026-01-19
**状态**: 设计提案

# 模块 API 参考

本文档包含所有核心模块的代码接口和实现细节。

## 1. 游戏状态识别 (game_state.py)

```python
class GameState:
    """游戏状态识别器"""

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
        """
        从屏幕截图更新游戏状态

        Args:
            screenshot: 屏幕截图 (numpy array)
        """
        # OCR 识别血量、金币
        # 目标检测识别位置
        pass

    def to_tensor(self):
        """
        转换为神经网络输入张量

        Returns:
            torch.Tensor: 游戏状态张量
        """
        pass

    def get_hero_position(self):
        """获取英雄位置"""
        return self.hero_position

    def get_health(self):
        """获取英雄血量"""
        return self.hero_health

    def is_in_danger(self):
        """判断是否处于危险状态"""
        # 根据敌方位置、血量等判断
        pass
```

## 2. 操作执行器 (action_executor.py)

```python
class ActionExecutor:
    """操作执行器"""

    def __init__(self, human_behavior=None):
        self.human_behavior = human_behavior

    def move_to(self, target_pos):
        """
        移动到目标位置（模拟人类速度）

        Args:
            target_pos: (x, y) 目标位置
        """
        # 计算移动时间
        # 模拟人类鼠标轨迹
        # 执行移动
        pass

    def attack_target(self, target_pos):
        """
        攻击目标

        Args:
            target_pos: (x, y) 目标位置
        """
        # 右键点击
        pass

    def cast_skill(self, skill_key, target_pos=None):
        """
        释放技能

        Args:
            skill_key: 技能按键 (Q/W/E/R/D/F)
            target_pos: (x, y) 目标位置，如果非指向性技能则为 None
        """
        # 按下技能键
        # 如果是指向性技能，移动鼠标到目标位置
        pass

    def use_heal(self):
        """使用治疗道具"""
        pass
```

## 3. AI 决策引擎 (ai_engine.py)

```python
class AIEngine:
    """AI决策引擎"""

    def __init__(self, model_path, hero_type):
        """
        初始化 AI 引擎

        Args:
            model_path: 模型文件路径
            hero_type: 英雄类型 (tank/mage/marksman/assassin/support)
        """
        self.model = self.load_model(model_path)
        self.hero_type = hero_type

    def load_model(self, model_path):
        """加载模型"""
        pass

    def decide_action(self, game_state):
        """
        根据游戏状态决策下一步操作

        Args:
            game_state: GameState 对象

        Returns:
            dict: 操作指令
        """
        action = self.model.predict(game_state.to_tensor())
        return action

    def update_policy(self, reward):
        """
        根据奖励更新策略（强化学习模式）

        Args:
            reward: 奖励值
        """
        pass
```

## 4. 录像解析 (replay_parser.py)

```python
import lol_replay_parser

class ReplayParser:
    """录像解析器"""

    def parse_replay(self, replay_path):
        """
        解析 .rofl 文件

        Args:
            replay_path: .rofl 文件路径

        Returns:
            parsed_data: 解析后的数据
        """
        replay = lol_replay_parser.parse(replay_path)
        return replay

    def extract_frames(self, fps=30):
        """
        提取关键帧

        Args:
            fps: 帧率

        Returns:
            frames: 帧列表
        """
        pass

    def extract_actions(self):
        """
        提取玩家操作序列

        Returns:
            actions: 操作序列
        """
        pass
```

## 5. 数据提取 (state_extractor.py & action_extractor.py)

### 状态提取器

```python
class StateExtractor:
    """状态特征提取器"""

    def extract(self, frame):
        """
        从帧中提取状态特征

        Args:
            frame: 游戏画面帧

        Returns:
            state: 状态特征向量
        """
        pass

    def normalize(self, state):
        """
        归一化状态

        Args:
            state: 原始状态

        Returns:
            normalized_state: 归一化后的状态
        """
        pass
```

### 动作提取器

```python
class ActionExtractor:
    """动作特征提取器"""

    def extract(self, frame_data):
        """
        从帧数据中提取动作

        Args:
            frame_data: 帧数据

        Returns:
            action: 动作
        """
        pass

    def encode(self, action):
        """
        编码动作

        Args:
            action: 原始动作

        Returns:
            encoded_action: 编码后的动作
        """
        pass
```

## 6. 数据加载器 (data_loader.py)

```python
import torch
from torch.utils.data import Dataset, DataLoader

class LoLDataset(Dataset):
    """英雄联盟数据集"""

    def __init__(self, data_path, augment=False):
        """
        初始化数据集

        Args:
            data_path: 数据路径
            augment: 是否进行数据增强
        """
        self.data_path = data_path
        self.augment = augment
        self.data = self.load_data()

    def load_data(self):
        """加载数据"""
        pass

    def __len__(self):
        """数据集大小"""
        return len(self.data)

    def __getitem__(self, idx):
        """获取数据项"""
        item = self.data[idx]

        if self.augment:
            item = self.augment_data(item)

        return item

    def augment_data(self, item):
        """
        数据增强

        Args:
            item: 数据项

        Returns:
            augmented_item: 增强后的数据
        """
        # 镜像翻转
        # 位置平移
        # 时间扭曲
        pass

def get_dataloader(data_path, batch_size=32, shuffle=True, num_workers=4):
    """
    获取数据加载器

    Args:
        data_path: 数据路径
        batch_size: 批量大小
        shuffle: 是否打乱
        num_workers: 工作线程数

    Returns:
        dataloader: 数据加载器
    """
    dataset = LoLDataset(data_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return dataloader
```

## 7. 策略模型 (model.py)

### 个人用户优化版（小模型）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoLAIModel(nn.Module):
    """英雄联盟AI策略网络（小模型 - 针对RTX5060 8GB优化）"""

    def __init__(self, num_actions=32):
        """
        初始化模型

        Args:
            num_actions: 动作数量（32种：8方向移动 + 普攻 + 技能等）
        """
        super().__init__()

        # CNN特征提取（小型）
        self.conv1 = nn.Conv2d(12, 32, kernel_size=8, stride=4)  # 4帧RGB堆叠 = 12通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)

        # LSTM序列处理
        self.lstm_hidden_size = 128
        self.lstm = nn.LSTM(128 * 7 * 7, self.lstm_hidden_size, batch_first=True)

        # 策略头（动作分类）
        self.policy_head = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

        # Dropout
        self.dropout = nn.Dropout(0.3)

    def forward(self, state):
        """
        前向传播

        Args:
            state: 输入状态 (batch, seq_len, C, H, W)

        Returns:
            action_probs: 动作概率分布
        """
        batch_size, seq_len = state.size(0), state.size(1)
        state = state.view(batch_size * seq_len, *state.size()[2:])

        # CNN 特征提取
        x = F.relu(self.conv1(state))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        visual_features = x.view(batch_size, seq_len, -1)

        # LSTM 序列建模
        lstm_out, _ = self.lstm(visual_features)
        lstm_features = lstm_out[:, -1, :]  # 取最后一个时间步

        # 全连接层
        x = torch.cat([lstm_features, visual_features[:, -1, :]], dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        action_logits = self.fc3(x)

        action_probs = F.softmax(action_logits, dim=1)

        return action_probs

    def predict(self, state):
        """
        预测动作

        Args:
            state: 输入状态

        Returns:
            action: 预测的动作
        """
        with torch.no_grad():
            action_probs = self.forward(state)
            action = torch.argmax(action_probs, dim=1)
        return action
```

## 8. 训练器 (trainer.py)

### 预训练器（行为克隆）

```python
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class BehaviorCloningTrainer:
    """行为克隆训练器"""

    def __init__(self, model, dataloader, lr=0.001, device='cuda'):
        """
        初始化训练器

        Args:
            model: 策略模型
            dataloader: 数据加载器
            lr: 学习率
            device: 设备
        """
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

    def train_epoch(self):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0

        for batch in tqdm(self.dataloader):
            states, actions = batch
            states, actions = states.to(self.device), actions.to(self.device)

            # 前向传播
            action_probs = self.model(states)

            # 计算损失
            loss = self.criterion(action_probs, actions)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.dataloader)
        return avg_loss

    def train(self, num_epochs):
        """
        训练模型

        Args:
            num_epochs: 训练轮数
        """
        for epoch in range(num_epochs):
            loss = self.train_epoch()
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")

            # 保存检查点
            if (epoch + 1) % 10 == 0:
                torch.save(self.model.state_dict(),
                          f"model_checkpoint_epoch_{epoch + 1}.pth")

    def save_model(self, path):
        """保存模型"""
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        """加载模型"""
        self.model.load_state_dict(torch.load(path))
```

### 强化学习训练器

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

class RLTrainer:
    """强化学习训练器"""

    def __init__(self, env, model_type='PPO'):
        """
        初始化训练器

        Args:
            env: 游戏环境
            model_type: 模型类型 (PPO/DQN)
        """
        self.env = env
        self.model_type = model_type

    def train(self, total_timesteps=100000):
        """
        训练智能体

        Args:
            total_timesteps: 总训练步数
        """
        if self.model_type == 'PPO':
            self.model = PPO("MlpPolicy", self.env, verbose=1)
        elif self.model_type == 'DQN':
            from stable_baselines3 import DQN
            self.model = DQN("MlpPolicy", self.env, verbose=1)

        self.model.learn(total_timesteps=total_timesteps)

    def save_model(self, path):
        """保存模型"""
        self.model.save(path)

    def load_model(self, path):
        """加载模型"""
        if self.model_type == 'PPO':
            self.model = PPO.load(path)
        elif self.model_type == 'DQN':
            self.model = DQN.load(path)
```

## 9. 大乱斗环境 (arena_env.py)

```python
import gymnasium as gym
import numpy as np

class ArenaEnv(gym.Env):
    """极地大乱斗环境"""

    def __init__(self):
        """初始化环境"""
        super().__init__()

        # 动作空间: [move_x, move_y, attack, skill_q, skill_w, skill_e, skill_r]
        self.action_space = gym.spaces.Box(
            low=np.array([-1, -1, 0, 0, 0, 0, 0]),
            high=np.array([1, 1, 1, 1, 1, 1, 1]),
            dtype=np.float32
        )

        # 状态空间: [hero_pos(2), hero_health(1), hero_mana(1),
        #             skills_cd(4), enemy_positions(10*2), ...]
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(100,),
            dtype=np.float32
        )

        self.current_state = None
        self.game_state = None

    def reset(self, seed=None, options=None):
        """
        重置环境

        Returns:
            observation: 初始观测
            info: 信息字典
        """
        super().reset(seed=seed)

        # 重置游戏状态
        self.game_state = GameState()
        self.current_state = self.game_state.to_tensor()

        info = {}

        return self.current_state, info

    def step(self, action):
        """
        执行动作

        Args:
            action: 动作向量

        Returns:
            observation: 新观测
            reward: 奖励
            terminated: 是否终止
            truncated: 是否截断
            info: 信息字典
        """
        # 执行动作
        self.execute_action(action)

        # 更新游戏状态
        self.game_state.update_from_screen(capture_screen())
        self.current_state = self.game_state.to_tensor()

        # 计算奖励
        reward = self.calculate_reward()

        # 检查是否结束
        terminated, truncated = self.check_done()

        info = {}

        return self.current_state, reward, terminated, truncated, info

    def execute_action(self, action):
        """执行动作"""
        pass

    def calculate_reward(self):
        """计算奖励"""
        # 击杀敌人: +1
        # 受到伤害: -0.1
        # 造成伤害: +0.01
        pass

    def check_done(self):
        """检查游戏是否结束"""
        pass
```

## 10. 强化学习智能体 (agent.py)

```python
class RLAgent:
    """强化学习智能体"""

    def __init__(self, env, model_path=None):
        """
        初始化智能体

        Args:
            env: 游戏环境
            model_path: 模型路径（可选）
        """
        self.env = env
        self.model = None

        if model_path:
            self.load_model(model_path)

    def train(self, num_episodes):
        """
        训练智能体

        Args:
            num_episodes: 训练回合数
        """
        pass

    def act(self, state):
        """
        根据状态选择动作

        Args:
            state: 当前状态

        Returns:
            action: 选择的动作
        """
        if self.model:
            action, _ = self.model.predict(state, deterministic=True)
        else:
            # 随机动作
            action = self.env.action_space.sample()

        return action
```

## 11. 屏幕截取 (screen_capture.py)

```python
import cv2
import numpy as np
import pygetwindow as gw
from PIL import ImageGrab

class ScreenCapture:
    """屏幕截取器"""

    def __init__(self, window_name="League of Legends"):
        """
        初始化截取器

        Args:
            window_name: 窗口名称
        """
        self.window_name = window_name
        self.window = None

    def find_window(self):
        """查找游戏窗口"""
        try:
            self.window = gw.getWindowsWithTitle(self.window_name)[0]
            return True
        except IndexError:
            return False

    def capture_full_screen(self):
        """
        截取完整屏幕

        Returns:
            screenshot: 屏幕截图
        """
        if not self.window:
            self.find_window()

        if self.window:
            # 截取指定窗口
            screenshot = ImageGrab.grab(bbox=(
                self.window.left,
                self.window.top,
                self.window.right,
                self.window.bottom
            ))
        else:
            # 截取全屏
            screenshot = ImageGrab.grab()

        screenshot = np.array(screenshot)
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)

        return screenshot

    def capture_region(self, region):
        """
        截取指定区域

        Args:
            region: (x1, y1, x2, y2) 区域坐标

        Returns:
            screenshot: 区域截图
        """
        screenshot = self.capture_full_screen()
        region_screenshot = screenshot[region[1]:region[3], region[0]:region[2]]

        return region_screenshot
```

## 12. 图像识别 (image_recognition.py)

```python
import cv2
import numpy as np
from paddleocr import PaddleOCR
from ultralytics import YOLO

class ImageRecognition:
    """图像识别器"""

    def __init__(self):
        """初始化识别器"""
        self.ocr = PaddleOCR(use_angle_cls=True, lang='ch')
        self.yolo = YOLO('yolov8n.pt')  # 需要训练或下载预训练模型

    def recognize_text(self, image):
        """
        OCR 识别文字

        Args:
            image: 输入图像

        Returns:
            texts: 识别的文字列表
        """
        result = self.ocr.ocr(image, cls=True)
        texts = [item[1][0] for item in result[0]]
        return texts

    def recognize_health(self, image):
        """
        识别血量数值

        Args:
            image: 血量区域图像

        Returns:
            health: 血量数值
        """
        text = self.recognize_text(image)
        if text:
            health = int(text[0].replace(',', ''))
            return health
        return None

    def detect_objects(self, image, classes=['hero', 'minion', 'tower']):
        """
        目标检测

        Args:
            image: 输入图像
            classes: 检测类别

        Returns:
            detections: 检测结果列表
        """
        results = self.yolo(image, classes=classes)
        return results

    def track_positions(self, detections):
        """
        追踪位置

        Args:
            detections: 检测结果

        Returns:
            positions: 位置列表 [(x, y), ...]
        """
        positions = []
        for result in detections:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                positions.append((int(center_x), int(center_y)))

        return positions
```

## 13. 输入模拟 (input_simulator.py)

```python
import pyautogui
import random

class InputSimulator:
    """输入模拟器"""

    def __init__(self):
        """初始化模拟器"""
        pyautogui.FAILSAFE = True

    def move_mouse(self, x, y, duration=None):
        """
        移动鼠标

        Args:
            x: 目标 x 坐标
            y: 目标 y 坐标
            duration: 移动时长（秒）
        """
        if duration is None:
            duration = random.uniform(0.1, 0.3)

        pyautogui.moveTo(x, y, duration=duration)

    def click(self, x=None, y=None, button='left'):
        """
        点击鼠标

        Args:
            x: x 坐标（可选）
            y: y 坐标（可选）
            button: 按键（left/right/middle）
        """
        if x is not None and y is not None:
            pyautogui.click(x, y, button=button)
        else:
            pyautogui.click(button=button)

    def press_key(self, key):
        """
        按下键盘按键

        Args:
            key: 按键
        """
        pyautogui.press(key)

    def hotkey(self, *keys):
        """
        组合键

        Args:
            *keys: 按键列表
        """
        pyautogui.hotkey(*keys)

    def drag_to(self, x, y, duration=None):
        """
        拖拽

        Args:
            x: 目标 x 坐标
            y: 目标 y 坐标
            duration: 拖拽时长（秒）
        """
        if duration is None:
            duration = random.uniform(0.2, 0.4)

        pyautogui.dragTo(x, y, duration=duration)
```

## 14. 人类行为模拟 (human_behavior.py)

```python
import random
import time
import math
import numpy as np
from dataclasses import dataclass
from enum import Enum

class EmotionState(Enum):
    """情绪状态"""
    NORMAL = "normal"
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    TIRED = "tired"
    EXCITED = "excited"

@dataclass
class PlayerProfile:
    """玩家画像"""
    base_apm: int = 200
    base_reaction_time: float = 0.2  # 秒
    aggression: float = 0.5  # 0-1
    focus: float = 0.8  # 0-1
    fatigue_rate: float = 0.01  # 每分钟疲劳增长率

class HumanBehavior:
    """
    高级人类行为模拟器

    特性：
    - Catmull-Rom 样条曲线模拟鼠标轨迹（更平滑）
    - Perlin Noise 添加自然随机性
    - 疲劳模型：随时间推移增加反应时间
    - 情绪模型：激进/保守状态影响行为
    - 上下文感知：根据游戏状态调整
    - 微动作：模拟真实玩家的微小动作
    """

    def __init__(self, profile: PlayerProfile = None):
        """
        初始化模拟器

        Args:
            profile: 玩家画像，如果为 None 则随机生成
        """
        self.profile = profile or self._generate_random_profile()

        # 状态变量
        self.current_emotion = EmotionState.NORMAL
        self.fatigue = 0.0  # 0-1
        self.session_start_time = time.time()
        self.action_count = 0
        self.kill_streak = 0  # 连杀
        self.death_streak = 0  # 连死

        # 鼠标历史（用于生成自然轨迹）
        self.mouse_history = []

    def _generate_random_profile(self) -> PlayerProfile:
        """生成随机玩家画像"""
        return PlayerProfile(
            base_apm=random.randint(150, 280),
            base_reaction_time=random.uniform(0.15, 0.3),
            aggression=random.uniform(0.3, 0.7),
            focus=random.uniform(0.7, 0.95),
            fatigue_rate=random.uniform(0.005, 0.02)
        )

    def update_fatigue(self):
        """更新疲劳度"""
        session_minutes = (time.time() - self.session_start_time) / 60
        self.fatigue = min(1.0, session_minutes * self.profile.fatigue_rate)

        # 根据疲劳度调整情绪
        if self.fatigue > 0.7:
            self.current_emotion = EmotionState.TIRED

    def update_emotion(self, game_state: dict):
        """
        根据游戏状态更新情绪

        Args:
            game_state: 游戏状态字典
        """
        # 连杀后变得更激进
        if self.kill_streak >= 3:
            self.current_emotion = EmotionState.EXCITED
        # 连死后变得更保守
        elif self.death_streak >= 2:
            self.current_emotion = EmotionState.CONSERVATIVE
        # 处于疲劳状态
        elif self.fatigue > 0.7:
            self.current_emotion = EmotionState.TIRED
        # 正常状态
        else:
            self.current_emotion = EmotionState.NORMAL

    def get_reaction_time(self) -> float:
        """
        获取当前反应时间（受疲劳和情绪影响）

        Returns:
            reaction_time: 反应时间（秒）
        """
        base = self.profile.base_reaction_time

        # 疲劳影响
        fatigue_factor = 1.0 + (self.fatigue * 0.5)

        # 情绪影响
        if self.current_emotion == EmotionState.EXCITED:
            emotion_factor = 0.8  # 激动时反应更快
        elif self.current_emotion == EmotionState.TIRED:
            emotion_factor = 1.5  # 疲劳时反应变慢
        else:
            emotion_factor = 1.0

        # 添加自然随机性
        random_factor = random.uniform(0.9, 1.1)

        return base * fatigue_factor * emotion_factor * random_factor

    def add_natural_delay(self, context: str = "normal"):
        """
        添加自然的操作延迟

        Args:
            context: 上下文（normal/combat/high_stress）
        """
        reaction_time = self.get_reaction_time()

        # 根据上下文调整
        if context == "high_stress":
            # 高压情况，反应更快
            delay = reaction_time * random.uniform(0.5, 0.8)
        elif context == "combat":
            # 战斗中
            delay = reaction_time * random.uniform(0.8, 1.0)
        else:
            # 正常情况
            delay = reaction_time

        # 添加微小的随机抖动（Perlin Noise 效果）
        jitter = random.gauss(0, 0.01)
        delay += jitter

        time.sleep(max(0, delay))

    def simulate_mouse_trajectory(self, start_pos, end_pos) -> list:
        """
        使用 Catmull-Rom 样条曲线生成自然的鼠标轨迹

        Args:
            start_pos: (x1, y1) 起始位置
            end_pos: (x2, y2) 结束位置

        Returns:
            points: 轨迹点列表 [(x, y), ...]
        """
        x1, y1 = start_pos
        x2, y2 = end_pos

        # 生成控制点
        num_points = random.randint(8, 15)
        points = self._generate_catmull_rom_trajectory(
            [start_pos, end_pos], num_points
        )

        # 添加 Perlin Noise
        points = self._add_perlin_noise(points)

        # 添加加速和减速效果
        points = self._apply_velocity_profile(points)

        return points

    def _generate_catmull_rom_trajectory(self, points: list, num_points: int) -> list:
        """
        生成 Catmull-Rom 样条曲线轨迹

        Args:
            points: 控制点列表
            num_points: 生成的轨迹点数

        Returns:
            trajectory: 轨迹点列表
        """
        if len(points) < 2:
            return points

        # 添加辅助点
        p0 = points[0]
        p1 = points[0]
        p2 = points[1]
        p3 = points[-1] if len(points) > 2 else points[1]

        trajectory = []

        for i in range(num_points):
            t = i / (num_points - 1)

            # Catmull-Rom 样条曲线公式
            t2 = t * t
            t3 = t2 * t

            x = 0.5 * ((2 * p1[0]) +
                       (-p0[0] + p2[0]) * t +
                       (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
                       (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)

            y = 0.5 * ((2 * p1[1]) +
                       (-p0[1] + p2[1]) * t +
                       (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
                       (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)

            trajectory.append((int(x), int(y)))

        return trajectory

    def _add_perlin_noise(self, points: list, intensity: float = 2.0) -> list:
        """
        添加 Perlin Noise（简化版）

        Args:
            points: 原始轨迹点
            intensity: 噪声强度

        Returns:
            noisy_points: 添加噪声后的轨迹点
        """
        noisy_points = []
        phase = 0

        for i, (x, y) in enumerate(points):
            # 简化的 Perlin Noise 模拟
            noise_x = intensity * math.sin(phase)
            noise_y = intensity * math.cos(phase)

            phase += 0.5  # 相位变化

            noisy_x = x + noise_x
            noisy_y = y + noise_y

            noisy_points.append((int(noisy_x), int(noisy_y)))

        return noisy_points

    def _apply_velocity_profile(self, points: list) -> list:
        """
        应用速度曲线（加速 -> 匀速 -> 减速）

        Args:
            points: 轨迹点列表

        Returns:
            velocity_points: 应用速度曲线后的点
        """
        n = len(points)
        if n < 3:
            return points

        # 速度曲线（S型）
        velocity_points = []
        for i in range(n):
            t = i / (n - 1)
            # S型曲线：0 -> 1 -> 0
            velocity = math.sin(t * math.pi)
            velocity_points.append(velocity)

        # 根据速度曲线调整点间距
        result = [points[0]]
        for i in range(1, n):
            # 根据速度决定是否添加这个点
            if random.random() < velocity_points[i]:
                result.append(points[i])

        return result

    def simulate_micro_movements(self):
        """模拟微小的鼠标抖动"""
        if random.random() < 0.03:  # 3% 概率
            # 微小抖动
            jitter_x = random.randint(-3, 3)
            jitter_y = random.randint(-3, 3)
            return jitter_x, jitter_y
        return 0, 0

    def simulate_useless_click(self) -> bool:
        """
        模拟无意义的点击

        Returns:
            should_click: 是否应该点击
        """
        # 焦虑或激动时更可能无意义点击
        if self.current_emotion in [EmotionState.AGGRESSIVE, EmotionState.EXCITED]:
            return random.random() < 0.05
        return random.random() < 0.01

    def should_make_mistake(self, context: str = "normal") -> tuple:
        """
        判断是否应该失误（受疲劳和情绪影响）

        Args:
            context: 上下文

        Returns:
            (should_mistake, mistake_type): 是否失误，失误类型
        """
        # 基础失误率
        base_error_rate = 0.02

        # 疲劳增加失误率
        fatigue_factor = self.fatigue * 0.08

        # 情绪影响
        if self.current_emotion == EmotionState.TIRED:
            emotion_factor = 0.05
        elif self.current_emotion == EmotionState.EXCITED:
            emotion_factor = 0.03  # 过度激动也可能失误
        else:
            emotion_factor = 0

        # 上下文影响
        if context == "high_stress":
            context_factor = 0.02
        else:
            context_factor = 0

        total_error_rate = base_error_rate + fatigue_factor + emotion_factor + context_factor

        if random.random() < total_error_rate:
            # 失误类型
            mistake_types = {
                "miss_skill": 0.3,      # 空技能
                "misposition": 0.3,     # 走位失误
                "wrong_target": 0.2,     # 错误目标
                "delay": 0.15,          # 延迟
                "cancel_action": 0.05   # 取消动作
            }

            # 根据情绪加权
            if self.current_emotion == EmotionState.TIRED:
                mistake_types["delay"] = 0.25
                mistake_types["cancel_action"] = 0.1

            # 选择失误类型
            mistake_type = random.choices(
                list(mistake_types.keys()),
                weights=list(mistake_types.values())
            )[0]

            return True, mistake_type

        return False, None

    def simulate_afk(self) -> float:
        """
        模拟短暂离开

        Returns:
            afk_duration: 离开时长（秒），0 表示不离开
        """
        # 疲劳时更可能短暂离开
        if self.fatigue > 0.6 and random.random() < 0.02:
            return random.uniform(5, 15)

        # 正常情况下的短暂离开（喝水等）
        if random.random() < 0.005:
            return random.uniform(3, 8)

        return 0

    def get_apm(self) -> int:
        """
        获取当前 APM（受疲劳影响）

        Returns:
            apm: 每分钟操作数
        """
        # 疲劳降低 APM
        fatigue_factor = 1.0 - (self.fatigue * 0.3)

        # 情绪影响
        if self.current_emotion == EmotionState.EXCITED:
            emotion_factor = 1.2
        elif self.current_emotion == EmotionState.TIRED:
            emotion_factor = 0.7
        else:
            emotion_factor = 1.0

        base_apm = self.profile.base_apm
        current_apm = int(base_apm * fatigue_factor * emotion_factor)

        # 添加随机波动
        current_apm += random.randint(-15, 15)

        return max(100, min(300, current_apm))

    def simulate_flick_movement(self, start_pos, target_pos) -> list:
        """
        模拟快速甩动鼠标（如快速瞄准）

        Args:
            start_pos: 起始位置
            target_pos: 目标位置

        Returns:
            points: 轨迹点列表
        """
        # 甩动移动更快，更直接
        points = self._generate_catmull_rom_trajectory(
            [start_pos, target_pos],
            num_points=random.randint(4, 6)
        )

        # 甩动时噪声更小
        points = self._add_perlin_noise(points, intensity=1.0)

        return points

    def simulate_precise_movement(self, start_pos, target_pos) -> list:
        """
        模拟精细操作（如点击小目标）

        Args:
            start_pos: 起始位置
            target_pos: 目标位置

        Returns:
            points: 轨迹点列表
        """
        # 精细操作更慢，更平滑
        points = self._generate_catmull_rom_trajectory(
            [start_pos, target_pos],
            num_points=random.randint(12, 18)
        )

        # 精细操作时在目标附近微调
        last_point = points[-1]
        fine_adjust_x = random.randint(-2, 2)
        fine_adjust_y = random.randint(-2, 2)
        points[-1] = (last_point[0] + fine_adjust_x, last_point[1] + fine_adjust_y)

        return points

    def record_action(self):
        """记录一次操作"""
        self.action_count += 1

    def record_kill(self):
        """记录击杀"""
        self.kill_streak += 1
        self.death_streak = 0
        self.update_emotion({})

    def record_death(self):
        """记录死亡"""
        self.death_streak += 1
        self.kill_streak = 0
        self.update_emotion({})
```

## 15. 录像转换器 (replay_converter.py)

```python
import json
import h5py
import numpy as np
from .replay_parser import ReplayParser
from .state_extractor import StateExtractor
from .action_extractor import ActionExtractor

class ReplayConverter:
    """录像转换器"""

    def __init__(self):
        """初始化转换器"""
        self.parser = ReplayParser()
        self.state_extractor = StateExtractor()
        self.action_extractor = ActionExtractor()

    def convert_single_replay(self, replay_path, output_path):
        """
        转换单个录像

        Args:
            replay_path: 录像文件路径
            output_path: 输出文件路径 (.h5)
        """
        # 解析录像
        replay_data = self.parser.parse_replay(replay_path)

        # 提取帧
        frames = self.parser.extract_frames(fps=30)

        # 提取状态和动作
        states = []
        actions = []

        for frame_data in frames:
            state = self.state_extractor.extract(frame_data['frame'])
            action = self.action_extractor.extract(frame_data)

            states.append(state)
            actions.append(action)

        # 保存为 HDF5 格式
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('states', data=np.array(states))
            f.create_dataset('actions', data=np.array(actions))
            f.attrs['replay_name'] = replay_data.name

        print(f"转换完成: {replay_path} -> {output_path}")

    def convert_batch(self, replay_dir, output_dir):
        """
        批量转换录像

        Args:
            replay_dir: 录像文件目录
            output_dir: 输出文件目录
        """
        import os

        replay_files = [f for f in os.listdir(replay_dir) if f.endswith('.rofl')]

        for replay_file in replay_files:
            replay_path = os.path.join(replay_dir, replay_file)
            output_path = os.path.join(output_dir, replay_file.replace('.rofl', '.h5'))

            try:
                self.convert_single_replay(replay_path, output_path)
            except Exception as e:
                print(f"转换失败: {replay_file}, 错误: {e}")
```

## 主程序入口 (main.py)

```python
import sys
from core.game_state import GameState
from core.action_executor import ActionExecutor
from core.ai_engine import AIEngine
from utils.screen_capture import ScreenCapture
from utils.image_recognition import ImageRecognition
from utils.human_behavior import HumanBehavior

def main():
    """主程序"""
    # 初始化组件
    screen_capture = ScreenCapture()
    image_recognition = ImageRecognition()
    human_behavior = HumanBehavior()
    game_state = GameState()
    action_executor = ActionExecutor(human_behavior)

    # 加载 AI 模型
    ai_engine = AIEngine(model_path="models/pretrained_model.pth", hero_type="mage")

    # 主循环
    while True:
        # 截取屏幕
        screenshot = screen_capture.capture_full_screen()

        # 更新游戏状态
        game_state.update_from_screen(screenshot)

        # AI 决策
        action = ai_engine.decide_action(game_state)

        # 执行动作（带人类行为模拟）
        if action['type'] == 'move':
            action_executor.move_to(action['target'])
        elif action['type'] == 'attack':
            action_executor.attack_target(action['target'])
        elif action['type'] == 'skill':
            action_executor.cast_skill(action['skill'], action.get('target'))

        # 随机延迟
        human_behavior.add_random_delay()

        # 偶尔休息
        if human_behavior.occasional_rest():
            continue

        # 检查是否应该失误
        should_mistake, mistake_type = human_behavior.random_mistake()
        if should_mistake:
            # 执行失误
            pass

if __name__ == '__main__':
    main()
```

## 相关文档

- [设计提案](design_proposal.md) - 整体设计和实施计划
- [架构设计](architecture.md) - 详细的技术架构和模块说明
- [防检测策略](anti_detection.md) - 完整的防检测实施方案

---

**文档版本**: 1.1
**最后更新**: 2026-01-20
**更新内容**：添加个人用户小模型架构示例

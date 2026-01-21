#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
推理脚本
实时游戏推理：屏幕捕获 -> 视觉编码 -> DQN决策 -> 操作执行
"""

import os
import sys
import time
import cv2
import numpy as np
import torch
from pathlib import Path
from threading import Thread
from collections import deque

import sys
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from ai.models.visual_encoder import VisualEncoder
from ai.models.dqn_agent import DQNAgent
from core.action_executor_with_rules import RuleBasedSafety
from utils.screen_capture import ScreenCapture
from utils.input_simulator import InputSimulator
from utils.paths import PROJECT_ROOT, MODEL_DIR
from utils.logger import logger


class InferenceConfig:
    """推理配置"""
    
    # 模型配置
    ENCODER_PATH = MODEL_DIR / "state_classifier" / "best.pth"
    DQN_PATH = MODEL_DIR / "dqn_agent" / "best.pth"
    
    # 图像配置
    TARGET_SIZE = (180, 320)  # (H, W)
    FRAME_SKIP = 4  # 每4帧处理一次
    
    # 设备配置
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 游戏窗口配置
    WINDOW_NAME = "League of Legends"
    WINDOW_WIDTH = 1920
    WINDOW_HEIGHT = 1080
    
    # 操作配置
    FPS_TARGET = 30  # 目标帧率
    APM_LIMIT = 250  # 最大APM
    MIN_DELAY = 0.2  # 最小延迟（秒）
    MAX_DELAY = 0.6  # 最大延迟（秒）
    
    # 日志配置
    LOG_ACTIONS = True
    LOG_INTERVAL = 10  # 每N个动作记录一次


class GameState:
    """游戏状态"""
    
    def __init__(self):
        self.hero_hp = 100.0
        self.hero_mana = 100.0
        self.gold = 0
        self.minions_killed = 0
        self.kills = 0
        self.assists = 0
        self.deaths = 0
        self.hero_position = (0, 0)
        self.minion_positions = []
        self.enemy_positions = []
        self.tower_position = (0, 0)


class InferenceEngine:
    """推理引擎"""
    
    def __init__(self, config=None):
        """
        Args:
            config: 推理配置
        """
        self.config = config or InferenceConfig()
        
        # 加载模型
        self._load_models()
        
        # 初始化组件
        self.screen_capture = ScreenCapture(self.config.WINDOW_NAME)
        self.input_simulator = InputSimulator()
        
        # 初始化安全规则检查器
        self.safety_checker = RuleBasedSafety()
        
        # 游戏状态
        self.game_state = GameState()
        
        # 帧队列（用于序列输入）
        self.frame_queue = deque(maxlen=4)
        
        # 统计信息
        self.action_count = 0
        self.action_history = []
        self.start_time = None
        
        # 运行状态
        self.is_running = False
        
        logger.info("推理引擎初始化完成")
    
    def _load_models(self):
        """加载模型"""
        logger.info("加载模型...")
        
        # 加载视觉编码器
        self.visual_encoder = VisualEncoder(
            input_channels=3,
            output_dim=256
        ).to(self.config.DEVICE)
        
        if self.config.ENCODER_PATH.exists():
            checkpoint = torch.load(self.config.ENCODER_PATH, map_location=self.config.DEVICE)
            # 从state_classifier checkpoint中提取encoder权重
            if 'model_state_dict' in checkpoint:
                # 需要适配checkpoint结构
                state_dict = {}
                for key, value in checkpoint['model_state_dict'].items():
                    if key.startswith('encoder.'):
                        new_key = key.replace('encoder.', '')
                        state_dict[new_key] = value
                self.visual_encoder.load_state_dict(state_dict)
            self.visual_encoder.eval()
            logger.info(f"视觉编码器已加载: {self.config.ENCODER_PATH}")
        else:
            logger.warning(f"视觉编码器不存在: {self.config.ENCODER_PATH}, 使用随机初始化")
        
        # 加载DQN智能体
        self.dqn_agent = DQNAgent(
            input_dim=256,
            action_dim=8,
            device=self.config.DEVICE
        )
        
        if self.config.DQN_PATH.exists():
            checkpoint = torch.load(self.config.DQN_PATH, map_location=self.config.DEVICE)
            self.dqn_agent.q_network.load_state_dict(checkpoint['agent_state_dict'])
            self.dqn_agent.target_network.load_state_dict(checkpoint['agent_state_dict'])
            self.dqn_agent.eval_mode()
            logger.info(f"DQN智能体已加载: {self.config.DQN_PATH}")
        else:
            logger.warning(f"DQN模型不存在: {self.config.DQN_PATH}, 使用随机初始化")
    
    def preprocess_frame(self, frame):
        """
        预处理帧
        
        Args:
            frame: (H, W, 3) BGR图像
        
        Returns:
            tensor: (3, H, W) 归一化张量
        """
        # 调整大小
        frame = cv2.resize(frame, (self.config.TARGET_SIZE[1], self.config.TARGET_SIZE[0]))
        
        # BGR转RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 转换为tensor并归一化
        frame = torch.from_numpy(frame).float() / 255.0
        frame = frame.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        
        # 标准化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        frame = (frame - mean) / std
        
        return frame
    
    def encode_frame(self, frame):
        """
        编码帧为特征向量
        
        Args:
            frame: (H, W, 3) BGR图像
        
        Returns:
            features: (256,) 特征向量
        """
        # 预处理
        tensor = self.preprocess_frame(frame)
        tensor = tensor.unsqueeze(0).to(self.config.DEVICE)  # (1, 3, H, W)
        
        # 编码
        with torch.no_grad():
            features = self.visual_encoder(tensor)
            features = features.squeeze(0).cpu().numpy()  # (256,)
        
        return features
    
    def select_action(self, features):
        """
        选择动作
        
        Args:
            features: (256,) 特征向量
        
        Returns:
            action: 动作索引
            action_name: 动作名称
            rule_triggered: 是否触发规则
        """
        # 转换为tensor
        tensor = torch.from_numpy(features).float().unsqueeze(0).to(self.config.DEVICE)
        
        # DQN选择动作
        with torch.no_grad():
            ai_action = self.dqn_agent.select_action(tensor, epsilon=0.0)
            ai_action = ai_action.item()
        
        # 安全规则检查
        state_dict = {
            'hero_hp': self.game_state.hero_hp,
            'hero_mana': self.game_state.hero_mana,
            'hero_position': self.game_state.hero_position,
            'teammate_count': len(self.game_state.minion_positions),  # 简化：用小兵数量
            'enemy_positions': self.game_state.enemy_positions,
            'teammate_positions': []  # TODO: 实现队友位置检测
        }
        
        final_action, rule_name = self.safety_checker.check_safety(state_dict, ai_action)
        
        action_name = DQNAgent.ACTION_MAP[final_action]
        
        return final_action, action_name, rule_name is not None
    
    def execute_action(self, action):
        """
        执行动作
        
        Args:
            action: 动作索引
        """
        action_name = DQNAgent.ACTION_MAP[action]
        
        if action_name.startswith('移动'):
            self._execute_move(action_name)
        elif action_name.startswith('攻击'):
            self._execute_attack(action_name)
        elif action_name == '回城':
            self._execute_recall()
        elif action_name == '等待':
            pass  # 不执行任何操作
    
    def _execute_move(self, direction):
        """
        执行移动动作
        
        Args:
            direction: 移动方向
        """
        # 计算移动方向
        screen_center_x = self.config.WINDOW_WIDTH // 2
        screen_center_y = self.config.WINDOW_HEIGHT // 2
        
        move_distance = 200
        
        if direction == '移动上':
            target_x, target_y = screen_center_x, screen_center_y - move_distance
        elif direction == '移动下':
            target_x, target_y = screen_center_x, screen_center_y + move_distance
        elif direction == '移动左':
            target_x, target_y = screen_center_x - move_distance, screen_center_y
        elif direction == '移动右':
            target_x, target_y = screen_center_x + move_distance, screen_center_y
        else:
            return
        
        # 右键移动
        self.input_simulator.right_click(target_x, target_y)
        
        logger.debug(f"移动: {direction} -> ({target_x}, {target_y})")
    
    def _execute_attack(self, target_type):
        """
        执行攻击动作
        
        Args:
            target_type: 目标类型（攻击小兵/攻击英雄）
        """
        # 简化实现：点击攻击键
        if target_type == '攻击小兵':
            # 寻找最近的小兵位置
            if self.game_state.minion_positions:
                target_pos = self.game_state.minion_positions[0]
                self.input_simulator.right_click(target_pos[0], target_pos[1])
                logger.debug(f"攻击小兵: {target_pos}")
        elif target_type == '攻击英雄':
            # 寻找最近的敌方英雄位置
            if self.game_state.enemy_positions:
                target_pos = self.game_state.enemy_positions[0]
                self.input_simulator.right_click(target_pos[0], target_pos[1])
                logger.debug(f"攻击英雄: {target_pos}")
    
    def _execute_recall(self):
        """执行回城"""
        # 按回城键（默认B）
        self.input_simulator.press_key('b')
        logger.debug("回城")
    
    def process_frame(self, frame):
        """
        处理单个帧
        
        Args:
            frame: (H, W, 3) BGR图像
        """
        # 编码帧
        features = self.encode_frame(frame)
        
        # 添加到帧队列
        self.frame_queue.append(features)
        
        # 队列满后选择动作
        if len(self.frame_queue) == self.frame_queue.maxlen:
            # 选择动作
            action, action_name = self.select_action(features)
            
            # 执行动作
            self.execute_action(action)
            
            # 记录
            self.action_count += 1
            self.action_history.append(action)
            
            if self.config.LOG_ACTIONS and self.action_count % self.config.LOG_INTERVAL == 0:
                logger.info(f"动作 {self.action_count}: {action_name}")
    
    def run(self):
        """运行推理循环"""
        logger.info("开始推理...")
        self.is_running = True
        self.start_time = time.time()
        
        frame_count = 0
        last_action_time = time.time()
        
        try:
            while self.is_running:
                # 捕获屏幕
                frame = self.screen_capture.capture()
                
                if frame is None:
                    logger.warning("未捕获到游戏窗口，等待重试...")
                    time.sleep(1.0)
                    continue
                
                frame_count += 1
                
                # 跳帧处理
                if frame_count % self.config.FRAME_SKIP == 0:
                    # 计算APM限制
                    current_time = time.time()
                    time_since_last_action = current_time - last_action_time
                    min_delay = self.config.MIN_DELAY
                    
                    if time_since_last_action >= min_delay:
                        # 处理帧
                        self.process_frame(frame)
                        last_action_time = current_time
                        
                        # 限制APM
                        delay = min(
                            max(time_since_last_action, min_delay),
                            self.config.MAX_DELAY
                        )
                        time.sleep(delay)
                    
                    # 计算FPS
                    elapsed = time.time() - self.start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    
                    if self.action_count % self.config.LOG_INTERVAL == 0:
                        logger.info(f"FPS: {fps:.1f}, 动作数: {self.action_count}")
        
        except KeyboardInterrupt:
            logger.info("接收到中断信号，停止推理")
        finally:
            self.stop()
    
    def stop(self):
        """停止推理"""
        self.is_running = False
        
        # 计算统计信息
        if self.start_time:
            elapsed = time.time() - self.start_time
            apm = (self.action_count / elapsed) * 60 if elapsed > 0 else 0
            
            logger.info("=" * 50)
            logger.info("推理统计:")
            logger.info(f"  运行时间: {elapsed:.2f}秒")
            logger.info(f"  总动作数: {self.action_count}")
            logger.info(f"  平均APM: {apm:.1f}")
            
            # 动作分布
            if self.action_history:
                from collections import Counter
                action_counts = Counter(self.action_history)
                logger.info(f"  动作分布:")
                for action, count in action_counts.most_common():
                    action_name = DQNAgent.ACTION_MAP[action]
                    logger.info(f"    {action_name}: {count} ({count/len(self.action_history)*100:.1f}%)")
            
            logger.info("=" * 50)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='DQN推理')
    parser.add_argument('--encoder', type=str, help='视觉编码器路径')
    parser.add_argument('--dqn', type=str, help='DQN模型路径')
    
    args = parser.parse_args()
    
    # 配置
    config = InferenceConfig()
    
    if args.encoder:
        config.ENCODER_PATH = Path(args.encoder)
    if args.dqn:
        config.DQN_PATH = Path(args.dqn)
    
    logger.info(f"推理配置:")
    logger.info(f"  视觉编码器: {config.ENCODER_PATH}")
    logger.info(f"  DQN模型: {config.DQN_PATH}")
    logger.info(f"  设备: {config.DEVICE}")
    
    # 创建推理引擎
    engine = InferenceEngine(config)
    
    # 运行推理
    engine.run()


if __name__ == '__main__':
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DQN离线训练脚本
使用高手录像训练DQN智能体
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import argparse
import numpy as np

import sys
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from ai.models.visual_encoder import VisualEncoder
from ai.models.dqn_agent import DQNAgent
from ai.models.replay_buffer_offline import OfflineReplayBuffer, Experience
from ai.finetune.arena_env_reward import CompositeReward, RewardConfig
from utils.paths import PROJECT_ROOT, MODEL_DIR, DATA_DIR
from utils.logger import logger


class DQNTrainingConfig:
    """DQN训练配置"""
    
    # 模型配置
    INPUT_DIM = 256  # 视觉编码器输出维度
    ACTION_DIM = 8  # 8个动作
    HIDDEN_DIMS = [128, 64, 32]  # 隐藏层维度
    DROPOUT = 0.5
    
    # 训练配置
    BATCH_SIZE = 32
    GAMMA = 0.99
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    TARGET_UPDATE_FREQ = 10
    MAX_EPISODES = 500
    TRAINING_STEPS_PER_EPISODE = 100
    
    # 离线训练配置
    REPLAY_BUFFER_SIZE = 5000
    EPSILON_START = 1.0
    EPSILON_END = 0.05
    EPSILON_DECAY = 0.995
    SHUFFLE_EPISODES = True  # 打乱episode顺序
    
    # 设备配置
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 数据配置
    DATA_DIR = DATA_DIR / "hero_states"
    
    # 保存配置
    SAVE_DIR = MODEL_DIR / "dqn_agent"
    SAVE_INTERVAL = 50  # 每N轮保存一次
    
    # TensorBoard配置
    LOG_DIR = PROJECT_ROOT / "logs" / "tensorboard" / "dqn_training"


class DQNTrainer:
    """DQN训练器"""
    
    def __init__(self, config, hero_names):
        self.config = config
        self.hero_names = hero_names
        self.device = config.DEVICE
        
        # 创建保存目录
        config.SAVE_DIR.mkdir(parents=True, exist_ok=True)
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        # 加载预训练编码器
        self.visual_encoder = VisualEncoder(
            input_channels=3,
            output_dim=config.INPUT_DIM
        ).to(self.device)
        
        # 冻结编码器
        self.visual_encoder.eval()
        for param in self.visual_encoder.parameters():
            param.requires_grad = False
        
        logger.info("视觉编码器已加载并冻结")
        
        # 创建DQN智能体
        self.agent = DQNAgent(
            input_dim=config.INPUT_DIM,
            action_dim=config.ACTION_DIM,
            hidden_dims=config.HIDDEN_DIMS,
            dropout=config.DROPOUT,
            learning_rate=config.LEARNING_RATE,
            gamma=config.GAMMA,
            target_update_freq=config.TARGET_UPDATE_FREQ,
            device=self.device
        )
        
        # 创建经验回放缓冲区
        self.replay_buffer = OfflineReplayBuffer(
            capacity=config.REPLAY_BUFFER_SIZE,
            frame_history=4
        )
        
        # 创建奖励函数
        self.reward_fn = CompositeReward(RewardConfig())
        
        # 创建TensorBoard
        self.writer = SummaryWriter(config.LOG_DIR)
        
        # 训练统计
        self.global_step = 0
        self.best_avg_reward = -float('inf')
        
        logger.info(f"DQN训练器初始化完成，设备: {self.device}")
        logger.info(f"英雄列表: {hero_names}")
    
    def load_replay_buffer(self):
        """加载经验回放缓冲区（从模拟数据或真实数据）"""
        logger.info("加载经验回放缓冲区...")
        
        # TODO: 从真实数据加载
        # 这里使用模拟数据作为示例
        self._generate_mock_data()
        
        stats = self.replay_buffer.get_statistics()
        logger.info(f"缓冲区加载完成: {stats}")
        
        # 打乱episode顺序
        if self.config.SHUFFLE_EPISODES:
            self.replay_buffer.shuffle_episodes()
    
    def _generate_mock_data(self):
        """生成模拟数据（用于测试）"""
        logger.warning("使用模拟数据，实际训练需要从真实录像加载")
        
        # 生成多个episode
        for episode_idx in range(20):
            episode_length = np.random.randint(50, 200)
            experiences = []
            
            state_features = np.random.randn(self.config.INPUT_DIM)
            
            for i in range(episode_length):
                action = np.random.randint(0, self.config.ACTION_DIM)
                next_state_features = np.random.randn(self.config.INPUT_DIM)
                reward = np.random.randn() * 0.5
                done = (i == episode_length - 1)
                
                exp = Experience(
                    state=state_features,
                    action=action,
                    reward=reward,
                    next_state=next_state_features,
                    done=done
                )
                experiences.append(exp)
                
                state_features = next_state_features
            
            self.replay_buffer.add_episode(experiences)
    
    def train_step(self):
        """训练一步"""
        # 从缓冲区采样
        try:
            batch = self.replay_buffer.sample(self.config.BATCH_SIZE)
        except ValueError:
            return None
        
        # 训练
        loss = self.agent.train_step(batch)
        
        return loss
    
    def train_episode(self, episode):
        """训练一个epoch"""
        self.agent.train_mode()
        
        episode_losses = []
        
        for step in tqdm(range(self.config.TRAINING_STEPS_PER_EPISODE),
                        desc=f"Episode {episode}",
                        leave=False):
            loss = self.train_step()
            
            if loss is not None:
                episode_losses.append(loss)
                
                # TensorBoard记录
                self.writer.add_scalar('train/loss', loss, self.global_step)
                self.global_step += 1
        
        avg_loss = np.mean(episode_losses) if episode_losses else 0.0
        
        # 计算平均奖励（从缓冲区统计）
        buffer_stats = self.replay_buffer.get_statistics()
        avg_reward = buffer_stats['avg_reward']
        
        # TensorBoard记录
        self.writer.add_scalar('train/episode_loss', avg_loss, episode)
        self.writer.add_scalar('train/avg_reward', avg_reward, episode)
        self.writer.add_scalar('train/buffer_usage', buffer_stats['buffer_usage'], episode)
        
        return avg_loss, avg_reward
    
    def train(self):
        """训练循环"""
        logger.info("开始训练")
        
        # 加载缓冲区
        self.load_replay_buffer()
        
        # 训练循环
        for episode in range(1, self.config.MAX_EPISODES + 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"Episode {episode}/{self.config.MAX_EPISODES}")
            logger.info(f"{'='*50}")
            
            # 训练一个episode
            avg_loss, avg_reward = self.train_episode(episode)
            
            # 日志
            logger.info(f"平均损失: {avg_loss:.4f}")
            logger.info(f"平均奖励: {avg_reward:.4f}")
            
            # 保存模型
            is_best = avg_reward > self.best_avg_reward
            if is_best:
                self.best_avg_reward = avg_reward
            
            self.save_checkpoint(episode, avg_reward, is_best)
        
        logger.info(f"\n训练完成，最佳平均奖励: {self.best_avg_reward:.4f}")
        self.writer.close()
    
    def save_checkpoint(self, episode, avg_reward, is_best=False):
        """保存检查点"""
        checkpoint = {
            'episode': episode,
            'agent_state_dict': self.agent.q_network.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'avg_reward': avg_reward,
            'best_avg_reward': self.best_avg_reward,
            'global_step': self.global_step,
            'config': self.config.__dict__
        }
        
        # 保存最新模型
        checkpoint_path = self.config.SAVE_DIR / "latest.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = self.config.SAVE_DIR / "best.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"保存最佳模型: {best_path}, 平均奖励: {avg_reward:.4f}")
        
        # 定期保存
        if episode % self.config.SAVE_INTERVAL == 0:
            epoch_path = self.config.SAVE_DIR / f"episode_{episode}.pth"
            torch.save(checkpoint, epoch_path)
            logger.info(f"保存检查点: {epoch_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='DQN离线训练')
    parser.add_argument('--heroes', type=str, nargs='+', required=True, help='英雄名称列表')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--episodes', type=int, default=500, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--resume', type=str, help='从检查点恢复训练')
    
    args = parser.parse_args()
    
    # 配置
    config = DQNTrainingConfig()
    config.BATCH_SIZE = args.batch_size
    config.MAX_EPISODES = args.episodes
    config.LEARNING_RATE = args.lr
    
    logger.info(f"训练配置: BATCH_SIZE={config.BATCH_SIZE}, MAX_EPISODES={config.MAX_EPISODES}")
    
    # 创建训练器
    trainer = DQNTrainer(config, args.heroes)
    
    # 恢复训练
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=config.DEVICE)
        trainer.agent.q_network.load_state_dict(checkpoint['agent_state_dict'])
        trainer.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.global_step = checkpoint['global_step']
        start_episode = checkpoint['episode'] + 1
        logger.info(f"从episode {start_episode}恢复训练")
    else:
        start_episode = 1
    
    # 训练
    trainer.train()


if __name__ == '__main__':
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
离线经验回放缓冲区
用于从高手录像提取和存储经验
"""

import random
import json
import numpy as np
from collections import deque
from pathlib import Path

import sys
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))

from utils.logger import logger


class Experience:
    """
    单个经验
    
    包含状态、动作、奖励、下一状态、是否终止
    """
    
    def __init__(self, state, action, reward, next_state, done, info=None):
        self.state = state  # (256,) 视觉特征向量
        self.action = action  # (1,) 动作索引
        self.reward = reward  # (1,) 标量奖励
        self.next_state = next_state  # (256,) 下一状态特征向量
        self.done = done  # (1,) 是否终止
        self.info = info or {}  # 额外信息
    
    def to_dict(self):
        """转换为字典"""
        return {
            'state': self.state.tolist() if isinstance(self.state, np.ndarray) else self.state,
            'action': int(self.action),
            'reward': float(self.reward),
            'next_state': self.next_state.tolist() if isinstance(self.next_state, np.ndarray) else self.next_state,
            'done': bool(self.done),
            'info': self.info
        }
    
    @staticmethod
    def from_dict(data):
        """从字典创建"""
        state = np.array(data['state'], dtype=np.float32)
        action = data['action']
        reward = data['reward']
        next_state = np.array(data['next_state'], dtype=np.float32)
        done = data['done']
        info = data.get('info', {})
        return Experience(state, action, reward, next_state, done, info)


class OfflineReplayBuffer:
    """
    离线经验回放缓冲区
    
    用于存储从高手录像提取的经验，支持批量采样
    """
    
    def __init__(self, capacity=5000, frame_history=4):
        """
        Args:
            capacity: 缓冲区容量
            frame_history: 帧历史长度（用于序列建模）
        """
        self.capacity = capacity
        self.frame_history = frame_history
        
        # 存储experience
        self.experiences = []
        self.episode_starts = []  # 记录每个episode的起始索引
        
        # 统计信息
        self.total_rewards = []
        self.episode_lengths = []
        
        logger.info(f"OfflineReplayBuffer初始化完成: 容量={capacity}, 帧历史={frame_history}")
    
    def add_experience(self, experience):
        """
        添加单个experience
        
        Args:
            experience: Experience对象
        """
        self.experiences.append(experience)
        
        # 如果缓冲区已满，移除最早的experience
        if len(self.experiences) > self.capacity:
            self.experiences.pop(0)
            
            # 更新episode_starts
            while self.episode_starts and self.episode_starts[0] > len(self.experiences):
                self.episode_starts.pop(0)
    
    def add_episode(self, experiences):
        """
        添加一个完整的episode
        
        Args:
            experiences: Experience对象列表
        """
        start_idx = len(self.experiences)
        
        for exp in experiences:
            self.add_experience(exp)
        
        # 记录episode起始位置
        if len(self.experiences) < self.capacity:
            self.episode_starts.append(start_idx)
        
        # 统计
        episode_reward = sum(exp.reward for exp in experiences)
        episode_length = len(experiences)
        
        self.total_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        
        logger.info(f"添加episode: 长度={episode_length}, 奖励={episode_reward:.2f}")
    
    def sample(self, batch_size):
        """
        随机采样一个batch
        
        Args:
            batch_size: 批次大小
        
        Returns:
            batch: (states, actions, rewards, next_states, dones)
        """
        if len(self.experiences) < batch_size:
            raise ValueError(f"缓冲区experience不足: {len(self.experiences)} < {batch_size}")
        
        indices = random.sample(range(len(self.experiences)), batch_size)
        batch = [self.experiences[i] for i in indices]
        
        # 转换为numpy数组
        states = np.array([exp.state for exp in batch], dtype=np.float32)
        actions = np.array([exp.action for exp in batch], dtype=np.int64)
        rewards = np.array([exp.reward for exp in batch], dtype=np.float32)
        next_states = np.array([exp.next_state for exp in batch], dtype=np.float32)
        dones = np.array([exp.done for exp in batch], dtype=np.float32)
        
        return (states, actions, rewards, next_states, dones)
    
    def sample_sequence(self, batch_size, seq_len=None):
        """
        采样序列（用于序列建模）
        
        Args:
            batch_size: 批次大小
            seq_len: 序列长度（默认使用frame_history）
        
        Returns:
            sequences: (states_seq, actions_seq, rewards_seq, next_states_seq, dones_seq)
        """
        if seq_len is None:
            seq_len = self.frame_history
        
        # 确保有足够的experience
        required_exp = seq_len * batch_size
        if len(self.experiences) < required_exp:
            raise ValueError(f"缓冲区experience不足: {len(self.experiences)} < {required_exp}")
        
        states_seq = []
        actions_seq = []
        rewards_seq = []
        next_states_seq = []
        dones_seq = []
        
        for _ in range(batch_size):
            # 随机选择一个episode
            episode_idx = random.choice(range(len(self.episode_starts)))
            start_idx = self.episode_starts[episode_idx]
            
            # 确定episode结束位置
            if episode_idx < len(self.episode_starts) - 1:
                end_idx = self.episode_starts[episode_idx + 1]
            else:
                end_idx = len(self.experiences)
            
            episode_length = end_idx - start_idx
            
            # 确保episode长度足够
            if episode_length < seq_len:
                continue
            
            # 随机选择序列起始位置
            seq_start_idx = start_idx + random.randint(0, episode_length - seq_len)
            
            # 提取序列
            seq = self.experiences[seq_start_idx:seq_start_idx + seq_len]
            
            states_seq.append(np.array([exp.state for exp in seq], dtype=np.float32))
            actions_seq.append(np.array([exp.action for exp in seq], dtype=np.int64))
            rewards_seq.append(np.array([exp.reward for exp in seq], dtype=np.float32))
            next_states_seq.append(np.array([exp.next_state for exp in seq], dtype=np.float32))
            dones_seq.append(np.array([exp.done for exp in seq], dtype=np.float32))
        
        # 转换为numpy数组
        states_seq = np.array(states_seq, dtype=np.float32)
        actions_seq = np.array(actions_seq, dtype=np.int64)
        rewards_seq = np.array(rewards_seq, dtype=np.float32)
        next_states_seq = np.array(next_states_seq, dtype=np.float32)
        dones_seq = np.array(dones_seq, dtype=np.float32)
        
        return (states_seq, actions_seq, rewards_seq, next_states_seq, dones_seq)
    
    def shuffle_episodes(self):
        """
        打乱episode顺序（用于离线训练模拟）
        
        这可以打破时间相关性，提高训练稳定性
        """
        # 按episode分组
        episodes = []
        for i in range(len(self.episode_starts)):
            start_idx = self.episode_starts[i]
            if i < len(self.episode_starts) - 1:
                end_idx = self.episode_starts[i + 1]
            else:
                end_idx = len(self.experiences)
            
            episode = self.experiences[start_idx:end_idx]
            episodes.append(episode)
        
        # 打乱episode顺序
        random.shuffle(episodes)
        
        # 重新构建experiences和episode_starts
        self.experiences = []
        self.episode_starts = []
        
        for episode in episodes:
            start_idx = len(self.experiences)
            self.experiences.extend(episode)
            self.episode_starts.append(start_idx)
        
        logger.info(f"已打乱{len(episodes)}个episode的顺序")
    
    def get_statistics(self):
        """获取统计信息"""
        if not self.total_rewards:
            return {
                'total_experiences': 0,
                'total_episodes': 0,
                'avg_reward': 0.0,
                'avg_length': 0.0,
                'buffer_usage': 0.0
            }
        
        stats = {
            'total_experiences': len(self.experiences),
            'total_episodes': len(self.episode_starts),
            'avg_reward': np.mean(self.total_rewards),
            'avg_length': np.mean(self.episode_lengths),
            'buffer_usage': len(self.experiences) / self.capacity
        }
        
        return stats
    
    def save(self, path):
        """保存缓冲区到文件"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'capacity': self.capacity,
            'frame_history': self.frame_history,
            'experiences': [exp.to_dict() for exp in self.experiences],
            'episode_starts': self.episode_starts,
            'total_rewards': self.total_rewards,
            'episode_lengths': self.episode_lengths
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"缓冲区已保存: {path}")
    
    def load(self, path):
        """从文件加载缓冲区"""
        path = Path(path)
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.capacity = data['capacity']
        self.frame_history = data['frame_history']
        self.experiences = [Experience.from_dict(exp) for exp in data['experiences']]
        self.episode_starts = data['episode_starts']
        self.total_rewards = data['total_rewards']
        self.episode_lengths = data['episode_lengths']
        
        logger.info(f"缓冲区已加载: {path}, 经验数={len(self.experiences)}, Episode数={len(self.episode_starts)}")
    
    def clear(self):
        """清空缓冲区"""
        self.experiences = []
        self.episode_starts = []
        self.total_rewards = []
        self.episode_lengths = []
        logger.info("缓冲区已清空")
    
    def __len__(self):
        return len(self.experiences)


def test_replay_buffer():
    """测试离线经验回放缓冲区"""
    print("=" * 50)
    print("测试离线经验回放缓冲区")
    print("=" * 50)
    
    # 创建缓冲区
    buffer = OfflineReplayBuffer(capacity=100, frame_history=4)
    
    # 创建虚拟episode
    def create_episode(length=10):
        experiences = []
        for i in range(length):
            state = np.random.randn(256)
            action = np.random.randint(0, 8)
            reward = np.random.randn()
            next_state = np.random.randn(256)
            done = (i == length - 1)
            exp = Experience(state, action, reward, next_state, done)
            experiences.append(exp)
        return experiences
    
    # 添加3个episode
    print("\n添加episode...")
    buffer.add_episode(create_episode(length=10))
    buffer.add_episode(create_episode(length=15))
    buffer.add_episode(create_episode(length=20))
    
    # 统计信息
    stats = buffer.get_statistics()
    print(f"\n统计信息: {stats}")
    
    # 测试采样
    print("\n测试批量采样...")
    states, actions, rewards, next_states, dones = buffer.sample(batch_size=4)
    print(f"states shape: {states.shape}")
    print(f"actions shape: {actions.shape}")
    print(f"rewards shape: {rewards.shape}")
    print(f"next_states shape: {next_states.shape}")
    print(f"dones shape: {dones.shape}")
    
    # 测试序列采样
    print("\n测试序列采样...")
    states_seq, actions_seq, rewards_seq, next_states_seq, dones_seq = buffer.sample_sequence(batch_size=2, seq_len=4)
    print(f"states_seq shape: {states_seq.shape}")
    print(f"actions_seq shape: {actions_seq.shape}")
    print(f"rewards_seq shape: {rewards_seq.shape}")
    print(f"next_states_seq shape: {next_states_seq.shape}")
    print(f"dones_seq shape: {dones_seq.shape}")
    
    # 测试打乱
    print("\n测试打乱episode...")
    buffer.shuffle_episodes()
    print("打乱完成")
    
    # 测试保存和加载
    print("\n测试保存和加载...")
    save_path = Path(__file__).parent.parent.parent / "data" / "test_buffer.json"
    buffer.save(save_path)
    
    new_buffer = OfflineReplayBuffer(capacity=100, frame_history=4)
    new_buffer.load(save_path)
    
    new_stats = new_buffer.get_statistics()
    print(f"加载后统计信息: {new_stats}")
    
    # 清理
    save_path.unlink()
    
    print("\n✅ 所有测试通过！")


if __name__ == '__main__':
    test_replay_buffer()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DQN网络架构
基于预训练视觉编码器的Deep Q-Network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

import sys
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))

from utils.logger import logger


class DQNNetwork(nn.Module):
    """
    DQN网络
    
    输入: (B, 256) - 视觉编码器输出的特征向量
    输出: (B, 8) - 8个动作的Q值
    
    参数量: ~45K
    """
    
    def __init__(self, input_dim=256, action_dim=8, hidden_dims=[128, 64, 32], dropout=0.5):
        super(DQNNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout
        
        # 构建隐藏层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # 输出层（Q值）
        self.q_head = nn.Linear(prev_dim, action_dim)
        
        # 初始化权重
        self._initialize_weights()
        
        logger.info(f"DQNNetwork初始化完成: 输入维度={input_dim}, 动作维度={action_dim}")
    
    def _initialize_weights(self):
        """Xavier初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, features):
        """
        前向传播
        
        Args:
            features: (B, 256) 特征向量
        
        Returns:
            q_values: (B, 8) 各动作的Q值
        """
        # 隐藏层
        x = self.hidden_layers(features)
        
        # Q值输出
        q_values = self.q_head(x)
        
        return q_values
    
    def select_action(self, features, epsilon=0.0):
        """
        选择动作（ε-greedy策略）
        
        Args:
            features: (B, 256) 特征向量
            epsilon: 探索率
        
        Returns:
            actions: (B,) 选择的动作
        """
        if epsilon > 0 and torch.rand(1).item() < epsilon:
            # 随机探索
            batch_size = features.size(0)
            actions = torch.randint(0, self.action_dim, (batch_size,), device=features.device)
            return actions
        else:
            # 贪婪策略（选择Q值最大的动作）
            q_values = self.forward(features)
            actions = torch.argmax(q_values, dim=1)
            return actions


class DQNAgent:
    """
    DQN智能体
    包含Q网络和目标网络
    """
    
    ACTION_MAP = {
        0: '移动上',
        1: '移动下',
        2: '移动左',
        3: '移动右',
        4: '攻击小兵',
        5: '攻击英雄',
        6: '回城',
        7: '等待'
    }
    
    REVERSE_ACTION_MAP = {v: k for k, v in ACTION_MAP.items()}
    
    def __init__(self, input_dim=256, action_dim=8, 
                 hidden_dims=[128, 64, 32], dropout=0.5,
                 learning_rate=0.001, gamma=0.99,
                 target_update_freq=10, device='cpu'):
        """
        Args:
            input_dim: 输入特征维度
            action_dim: 动作空间维度
            hidden_dims: 隐藏层维度
            dropout: Dropout率
            learning_rate: 学习率
            gamma: 折扣因子
            target_update_freq: 目标网络更新频率
            device: 设备
        """
        self.device = torch.device(device)
        self.action_dim = action_dim
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.update_count = 0
        
        # Q网络
        self.q_network = DQNNetwork(
            input_dim=input_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            dropout=dropout
        ).to(self.device)
        
        # 目标网络
        self.target_network = DQNNetwork(
            input_dim=input_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            dropout=dropout
        ).to(self.device)
        
        # 初始化目标网络
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 损失函数
        self.loss_fn = nn.SmoothL1Loss()
        
        logger.info(f"DQNAgent初始化完成，设备: {self.device}")
    
    def select_action(self, features, epsilon=0.0):
        """
        选择动作
        
        Args:
            features: (B, 256) 特征向量
            epsilon: 探索率
        
        Returns:
            actions: (B,) 选择的动作
        """
        self.q_network.eval()
        with torch.no_grad():
            actions = self.q_network.select_action(features, epsilon)
        return actions
    
    def train_step(self, batch):
        """
        训练一步
        
        Args:
            batch: 训练批次
                - states: (B, 256) 状态特征
                - actions: (B,) 动作
                - rewards: (B,) 奖励
                - next_states: (B, 256) 下一状态特征
                - dones: (B,) 是否终止
        
        Returns:
            loss: 损失值
        """
        states, actions, rewards, next_states, dones = batch
        
        # 转换为tensor（如果需要）
        if not isinstance(states, torch.Tensor):
            states = torch.FloatTensor(states)
        if not isinstance(actions, torch.Tensor):
            actions = torch.LongTensor(actions)
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.FloatTensor(rewards)
        if not isinstance(next_states, torch.Tensor):
            next_states = torch.FloatTensor(next_states)
        if not isinstance(dones, torch.Tensor):
            dones = torch.FloatTensor(dones)
        
        # 转换到设备
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # 计算当前Q值
        self.q_network.train()
        q_values = self.q_network(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 计算目标Q值
        self.target_network.eval()
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            next_q_value = next_q_values.max(1)[0]
            target_q_value = rewards + self.gamma * next_q_value * (1 - dones)
        
        # 计算损失
        loss = self.loss_fn(q_value, target_q_value)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        
        self.optimizer.step()
        
        # 更新目标网络
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.update_target_network()
        
        return loss.item()
    
    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        logger.debug(f"目标网络已更新（更新次数: {self.update_count}）")
    
    def save(self, path):
        """保存模型"""
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'update_count': self.update_count
        }
        torch.save(checkpoint, path)
        logger.info(f"模型已保存: {path}")
    
    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.update_count = checkpoint['update_count']
        logger.info(f"模型已加载: {path}")
    
    def eval_mode(self):
        """设置为评估模式"""
        self.q_network.eval()
        self.target_network.eval()
    
    def train_mode(self):
        """设置为训练模式"""
        self.q_network.train()
        self.target_network.eval()


def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_dqn():
    """测试DQN网络"""
    print("=" * 50)
    print("测试DQN网络")
    print("=" * 50)
    
    # 创建DQN网络
    dqn_net = DQNNetwork(input_dim=256, action_dim=8)
    
    # 统计参数量
    print(f"DQN网络参数量: {count_parameters(dqn_net):,}")
    
    # 测试前向传播
    batch_size = 4
    features = torch.randn(batch_size, 256)
    
    print(f"\n输入shape: {features.shape}")
    
    q_values = dqn_net(features)
    print(f"Q值输出shape: {q_values.shape}")
    print(f"Q值: {q_values}")
    
    # 测试动作选择
    actions_greedy = dqn_net.select_action(features, epsilon=0.0)
    actions_random = dqn_net.select_action(features, epsilon=1.0)
    
    print(f"\n贪婪策略动作: {actions_greedy}")
    print(f"随机策略动作: {actions_random}")
    
    # 测试DQN智能体
    print("\n" + "=" * 50)
    print("测试DQN智能体")
    print("=" * 50)
    
    agent = DQNAgent(
        input_dim=256,
        action_dim=8,
        learning_rate=0.001,
        device='cpu'
    )
    
    # 统计参数量
    total_params = count_parameters(agent.q_network)
    print(f"Q网络参数量: {total_params:,}")
    
    # 测试动作选择
    actions = agent.select_action(features, epsilon=0.1)
    print(f"\n智能体动作: {actions}")
    print(f"动作名称: {[DQNAgent.ACTION_MAP[a.item()] for a in actions]}")
    
    # 模拟训练步骤
    states = features
    actions = torch.randint(0, 8, (batch_size,))
    rewards = torch.randn(batch_size)
    next_states = torch.randn(batch_size, 256)
    dones = torch.zeros(batch_size)
    
    batch = (states, actions, rewards, next_states, dones)
    loss = agent.train_step(batch)
    
    print(f"\n训练损失: {loss:.4f}")
    print(f"目标网络更新次数: {agent.update_count}")
    
    print("\n✅ 所有测试通过！")


if __name__ == '__main__':
    test_dqn()

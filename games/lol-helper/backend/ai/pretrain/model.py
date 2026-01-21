import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class LoLAIModel(nn.Module):
    
    def __init__(self, num_actions: int = 32, hidden_size: int = 128):
        super().__init__()
        
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        
        # CNN特征提取（小型）
        # 输入: (batch, channels=12, height=180, width=320)
        self.cnn = nn.Sequential(
            # Conv1: 12 -> 32, kernel=8, stride=4
            # 输出: (batch, 32, 45, 80)
            nn.Conv2d(12, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.2),
            
            # Conv2: 32 -> 64, kernel=4, stride=2
            # 输出: (batch, 64, 22, 40)
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.2),
            
            # Conv3: 64 -> 128, kernel=3, stride=1
            # 输出: (batch, 128, 22, 40)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.2),
        )
        
        # CNN输出扁平化后的维度
        # 128 * 22 * 40 = 112640
        self.cnn_output_size = 128 * 22 * 40
        
        # LSTM序列处理
        self.lstm = nn.LSTM(
            self.cnn_output_size,
            hidden_size,
            batch_first=True,
            num_layers=2,
            dropout=0.2
        )
        
        # 策略头（动作分类）
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_actions)
        )
    
    def _init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """初始化LSTM隐藏状态"""
        h0 = torch.zeros(2, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(2, batch_size, self.hidden_size, device=device)
        return (h0, c0)
    
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch, seq_len, channels, height, width)
            hidden: LSTM隐藏状态 (可选）
        
        Returns:
            logits: 动作概率分布 (batch, seq_len, num_actions)
            hidden: LSTM新的隐藏状态
        """
        batch_size, seq_len, channels, height, width = x.shape
        
        # 重塑为 (batch * seq_len, channels, height, width) 进行CNN处理
        x_reshaped = x.reshape(batch_size * seq_len, channels, height, width)
        
        # CNN特征提取
        cnn_features = self.cnn(x_reshaped)
        
        # 扁平化
        cnn_features_flat = cnn_features.view(batch_size * seq_len, -1)
        
        # 重塑为 (batch, seq_len, cnn_features_flat) 进行LSTM处理
        cnn_features_seq = cnn_features_flat.view(batch_size, seq_len, -1)
        
        # LSTM处理
        if hidden is None:
            hidden = self._init_hidden(batch_size, x.device)
        
        lstm_output, hidden = self.lstm(cnn_features_seq, hidden)
        
        # 策略头
        logits = self.policy_head(lstm_output)
        
        return logits, hidden
    
    def predict(self, state: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        """
        预测动作（推理模式）
        
        Args:
            state: 单个状态或批次 (batch, channels, height, width)
            hidden: LSTM隐藏状态 (可选）
        
        Returns:
            action_logits: 动作logits (batch, num_actions)
            hidden: LSTM新的隐藏状态
        """
        # 如果输入是单个状态，添加batch维度和seq_len维度
        if state.dim() == 3:
            state = state.unsqueeze(0).unsqueeze(0)  # (1, 1, channels, height, width)
        elif state.dim() == 4:
            state = state.unsqueeze(1)  # (batch, 1, channels, height, width)
        
        logits, hidden = self.forward(state, hidden)
        
        # 返回最后一个时间步的输出
        action_logits = logits[:, -1, :]
        
        return action_logits, hidden
    
    def get_action(self, state: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple]:
        """
        获取动作（用于推理）
        
        Args:
            state: 状态 (batch, channels, height, width)
            hidden: LSTM隐藏状态
        
        Returns:
            action: 选择的动作 (batch,)
            hidden: LSTM新的隐藏状态
        """
        with torch.no_grad():
            logits, hidden = self.predict(state, hidden)
            probabilities = F.softmax(logits, dim=-1)
            action = torch.multinomial(probabilities, num_samples=1).squeeze(-1)
        
        return action, hidden


if __name__ == "__main__":
    print("测试LoLAIModel...")
    
    # 创建模型
    model = LoLAIModel(num_actions=32, hidden_size=128)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试前向传播
    batch_size = 4
    seq_len = 4
    channels = 12
    height = 180
    width = 320
    
    x = torch.randn(batch_size, seq_len, channels, height, width)
    logits, hidden = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {logits.shape}")
    
    # 测试预测
    state = torch.randn(batch_size, channels, height, width)
    action_logits, hidden = model.predict(state)
    print(f"预测输入形状: {state.shape}")
    print(f"预测输出形状: {action_logits.shape}")
    
    # 测试获取动作
    action, hidden = model.get_action(state)
    print(f"动作形状: {action.shape}")
    print(f"动作示例: {action}")
    
    print("\n模型测试完成！")

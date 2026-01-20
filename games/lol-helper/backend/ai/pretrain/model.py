import torch
import torch.nn as nn
import torch.nn.functional as F


class LoLAIModel(nn.Module):
    
    def __init__(self, num_actions: int = 32, hidden_size: int = 128):
        super().__init__()
        
        self.conv1 = nn.Conv2d(12, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        
        self.lstm_hidden_size = hidden_size
        self.lstm = nn.LSTM(128 * 7 * 7, hidden_size, batch_first=True)
        
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )
        
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
    
    def predict(self, state: torch.Tensor) -> torch.Tensor:
        pass

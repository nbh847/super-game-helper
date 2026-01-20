import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Any
import numpy as np
import h5py
from pathlib import Path
import json
from tqdm import tqdm


class LoLDataset(Dataset):
    """英雄联盟数据集
    
    从处理后的.h5文件加载数据
    """
    
    def __init__(self, data_path: str, augment: bool = False, 
                 frame_stack: int = 4):
        """
        初始化数据集
        
        Args:
            data_path: 数据目录或文件路径
            augment: 是否进行数据增强
            frame_stack: 堆叠帧数
        """
        self.data_path = Path(data_path)
        self.augment = augment
        self.frame_stack = frame_stack
        
        # 加载数据
        self.data = self.load_data()
        
        print(f"加载数据集: {len(self)} 个样本")
    
    def load_data(self) -> List[Tuple[Any, Any]]:
        """加载数据
        
        可以是单个.h5文件或目录中的多个.h5文件
        """
        data = []
        
        if self.data_path.is_file() and self.data_path.suffix == '.h5':
            # 单个文件
            data.extend(self._load_h5_file(self.data_path))
        elif self.data_path.is_dir():
            # 目录中的多个文件
            h5_files = list(self.data_path.glob('*.h5'))
            print(f"找到 {len(h5_files)} 个.h5文件")
            
            for h5_file in tqdm(h5_files, desc="加载数据"):
                file_data = self._load_h5_file(h5_file)
                data.extend(file_data)
        
        return data
    
    def _load_h5_file(self, h5_file: Path) -> List[Tuple[Any, Any]]:
        """加载单个.h5文件"""
        data = []
        
        try:
            with h5py.File(h5_file, 'r') as f:
                states = f['states'][:]  # shape: (num_samples, state_dim)
                actions = f['actions'][:]  # shape: (num_samples,)
                
                # 转换为样本列表
                for i in range(len(states)):
                    data.append((states[i], actions[i]))
                    
        except Exception as e:
            print(f"加载文件失败 {h5_file}: {e}")
        
        return data
    
    def __len__(self) -> int:
        """数据集大小"""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取数据项
        
        Args:
            idx: 索引
            
        Returns:
            (state, action): 状态和动作
        """
        state, action = self.data[idx]
        
        # 转换为numpy
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
        if not isinstance(action, (int, np.ndarray)):
            action = np.array(action, dtype=np.int64)
        
        # 数据增强
        if self.augment:
            state, action = self.augment_data(state, action)
        
        # 转换为tensor
        state_tensor = torch.FloatTensor(state)
        action_tensor = torch.LongTensor([action]) if isinstance(action, (int, np.integer)) else torch.LongTensor(action)
        
        return state_tensor, action_tensor
    
    def augment_data(self, state: np.ndarray, 
                      action: int) -> Tuple[np.ndarray, int]:
        """数据增强
        
        包括：镜像翻转、位置平移、时间扭曲等
        """
        # 随机选择增强方式
        augment_type = np.random.choice([
            'none', 'mirror', 'shift', 'noise'
        ], p=[0.6, 0.15, 0.15, 0.1])
        
        if augment_type == 'mirror':
            # 镜像翻转（水平）
            state, action = self._mirror_flip(state, action)
        elif augment_type == 'shift':
            # 位置平移
            state = self._spatial_shift(state)
        elif augment_type == 'noise':
            # 添加噪声
            state = self._add_noise(state)
        
        return state, action
    
    def _mirror_flip(self, state: np.ndarray, 
                      action: int) -> Tuple[np.ndarray, int]:
        """水平镜像翻转
        
        只翻转位置坐标，不影响其他特征
        """
        # 假设前两个特征是位置（x, y）
        # 只翻转x坐标
        new_state = state.copy()
        new_state[0] = 1.0 - new_state[0]  # 1 - x
        
        # 调整移动方向
        # 0↔2, 1↔3, 4↔6, 5↔7
        action_mapping = {
            0: 2, 2: 0,  # 上 ↔ 右
            1: 3, 3: 1,  # 右上 ↔ 右下
            4: 6, 6: 4,  # 下 ↔ 左
            5: 7, 7: 5   # 左下 ↔ 左上
        }
        if action in action_mapping:
            action = action_mapping[action]
        
        return new_state, action
    
    def _spatial_shift(self, state: np.ndarray) -> np.ndarray:
        """空间平移"""
        # 在小范围内随机平移位置
        shift_x = np.random.uniform(-0.05, 0.05)
        shift_y = np.random.uniform(-0.05, 0.05)
        
        new_state = state.copy()
        
        # 平移前两个位置特征
        new_state[0] = np.clip(new_state[0] + shift_x, 0, 1)
        new_state[1] = np.clip(new_state[1] + shift_y, 0, 1)
        
        return new_state
    
    def _add_noise(self, state: np.ndarray) -> np.ndarray:
        """添加高斯噪声"""
        noise = np.random.normal(0, 0.01, size=state.shape)
        new_state = state + noise
        
        # 限制在[0, 1]范围内
        new_state = np.clip(new_state, 0, 1)
        
        return new_state
    
    def _process_item(self, item: Tuple[Any, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """处理单个数据项（已弃用，使用__getitem__）"""
        state, action = item
        
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
        
        state_tensor = torch.FloatTensor(state)
        action_tensor = torch.LongTensor([action]) if isinstance(action, (int, np.integer)) else torch.LongTensor(action)
        
        return state_tensor, action_tensor


def get_dataloader(data_path: str, batch_size: int = 16, 
                    shuffle: bool = True, num_workers: int = 2, 
                    augment: bool = False, frame_stack: int = 4) -> DataLoader:
    """
    获取数据加载器
    
    Args:
        data_path: 数据路径
        batch_size: 批量大小
        shuffle: 是否打乱
        num_workers: 工作线程数
        augment: 是否数据增强
        frame_stack: 堆叠帧数
        
    Returns:
        dataloader: 数据加载器
    """
    dataset = LoLDataset(data_path, augment=augment, frame_stack=frame_stack)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True  # 加速数据传输到GPU
    )
    
    return dataloader


def split_dataset(data_path: str, train_ratio: float = 0.8,
                 seed: int = 42) -> Tuple[List[str], List[str]]:
    """
    划分数据集为训练集和验证集
    
    Args:
        data_path: 数据目录
        train_ratio: 训练集比例
        seed: 随机种子
        
    Returns:
        (train_files, val_files): 训练集和验证集文件列表
    """
    import random
    
    # 获取所有.h5文件
    data_dir = Path(data_path)
    h5_files = list(data_dir.glob('*.h5'))
    
    # 打乱文件列表
    random.seed(seed)
    random.shuffle(h5_files)
    
    # 划分
    split_idx = int(len(h5_files) * train_ratio)
    train_files = [str(f) for f in h5_files[:split_idx]]
    val_files = [str(f) for f in h5_files[split_idx:]]
    
    print(f"训练集: {len(train_files)} 个文件")
    print(f"验证集: {len(val_files)} 个文件")
    
    return train_files, val_files


def create_sample_dataset(output_path: str, num_samples: int = 1000):
    """
    创建示例数据集用于测试
    
    Args:
        output_path: 输出路径
        num_samples: 样本数量
    """
    import os
    
    os.makedirs(output_path, exist_ok=True)
    
    # 状态维度和动作数量
    state_dim = 128
    num_actions = 32
    
    # 生成随机数据
    states = np.random.rand(num_samples, state_dim).astype(np.float32)
    actions = np.random.randint(0, num_actions, size=num_samples).astype(np.int64)
    
    # 保存为.h5文件
    h5_path = Path(output_path) / "sample_dataset.h5"
    
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('states', data=states)
        f.create_dataset('actions', data=actions)
        f.attrs['num_samples'] = num_samples
        f.attrs['state_dim'] = state_dim
        f.attrs['num_actions'] = num_actions
    
    print(f"创建示例数据集: {h5_path}")
    print(f"样本数量: {num_samples}")

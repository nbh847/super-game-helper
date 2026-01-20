import json
import h5py
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import pickle
from tqdm import tqdm

from .replay_parser import ReplayParser
from .state_extractor import StateExtractor
from .action_extractor import ActionExtractor


class ReplayConverter:
    """录像转换器
    
    将.rofl文件转换为训练数据
    """
    
    def __init__(self):
        """初始化转换器"""
        self.parser = ReplayParser()
        self.state_extractor = StateExtractor()
        self.action_extractor = ActionExtractor()
        
        self.conversion_stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'failed_files': []
        }
    
    def convert_single_replay(self, replay_path: str, 
                              output_path: str, 
                              fps: int = 30,
                              skip_frames: int = 4) -> bool:
        """
        转换单个录像
        
        Args:
            replay_path: .rofl文件路径
            output_path: 输出文件路径（.h5）
            fps: 帧率
            skip_frames: 跳帧数（每隔N帧采样一次）
            
        Returns:
            success: 是否成功
        """
        self.conversion_stats['total'] += 1
        
        try:
            # 解析录像
            print(f"解析录像: {replay_path}")
            replay_data = self.parser.parse_replay(replay_path)
            
            if not replay_data:
                self.conversion_stats['failed'] += 1
                self.conversion_stats['failed_files'].append(replay_path)
                return False
            
            # 提取帧
            frames = self._extract_frames(replay_data, fps, skip_frames)
            
            if not frames:
                print(f"警告: 没有提取到帧数据")
                self.conversion_stats['failed'] += 1
                self.conversion_stats['failed_files'].append(replay_path)
                return False
            
            # 提取状态和动作
            states, actions = self._extract_states_and_actions(frames)
            
            if len(states) == 0 or len(actions) == 0:
                print(f"警告: 没有提取到状态或动作")
                self.conversion_stats['failed'] += 1
                self.conversion_stats['failed_files'].append(replay_path)
                return False
            
            # 确保states和actions长度一致
            min_len = min(len(states), len(actions))
            states = states[:min_len]
            actions = actions[:min_len]
            
            # 保存为HDF5格式
            self._save_to_h5(output_path, states, actions, replay_data.get('metadata', {}))
            
            print(f"转换成功: {output_path}")
            print(f"  - 样本数量: {len(states)}")
            print(f"  - 状态维度: {states[0].shape if len(states) > 0 else 0}")
            
            self.conversion_stats['success'] += 1
            return True
            
        except Exception as e:
            print(f"转换失败: {replay_path}, 错误: {e}")
            self.conversion_stats['failed'] += 1
            self.conversion_stats['failed_files'].append(replay_path)
            return False
    
    def _extract_frames(self, replay_data: Dict[str, Any], 
                        fps: int, skip_frames: int) -> List[Dict[str, Any]]:
        """提取关键帧"""
        metadata = replay_data.get('metadata', {})
        packets = replay_data.get('packets', [])
        
        duration = metadata.get('game_duration', 0)
        num_frames = int(duration * fps / skip_frames)
        
        frames = []
        
        # 简化版本：生成模拟帧数据
        # 实际实现需要从packets中解析
        for i in range(num_frames):
            frame = {
                'frame_number': i,
                'timestamp': i * skip_frames / fps,
                'hero_position': np.random.randint(0, 1920, size=2),
                'health': np.random.randint(1000, 5000),
                'mana': np.random.randint(500, 3000),
                'level': np.random.randint(1, 18),
                'gold': np.random.randint(0, 20000),
                'kills': np.random.randint(0, 20),
                'deaths': np.random.randint(0, 10),
                'assists': np.random.randint(0, 20),
                'skill_q_cd': np.random.choice([0, 1]),
                'skill_w_cd': np.random.choice([0, 1]),
                'skill_e_cd': np.random.choice([0, 1]),
                'skill_r_cd': np.random.choice([0, 1]),
                'enemies': self._generate_enemies(5),
                'minions': self._generate_minions(20),
                'tower': self._generate_tower()
            }
            frames.append(frame)
        
        return frames
    
    def _generate_enemies(self, num_enemies: int) -> List[Dict[str, Any]]:
        """生成敌方英雄"""
        enemies = []
        
        for i in range(num_enemies):
            enemy = {
                'id': i,
                'position': np.random.randint(0, 1920, size=2),
                'health': np.random.randint(1000, 5000),
                'is_enemy': True
            }
            enemies.append(enemy)
        
        return enemies
    
    def _generate_minions(self, num_minions: int) -> List[Dict[str, Any]]:
        """生成小兵"""
        minions = []
        
        for i in range(num_minions):
            minion = {
                'id': i,
                'position': np.random.randint(0, 1920, size=2),
                'health': np.random.randint(100, 500),
                'is_enemy': np.random.choice([True, False])
            }
            minions.append(minion)
        
        return minions
    
    def _generate_tower(self) -> Optional[Dict[str, Any]]:
        """生成防御塔"""
        if np.random.random() > 0.7:
            return None
        
        tower = {
            'position': np.random.randint(0, 1920, size=2),
            'health': np.random.randint(2000, 5000),
            'is_enemy': np.random.choice([True, False])
        }
        
        return tower
    
    def _extract_states_and_actions(self, 
                                   frames: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """提取状态和动作"""
        states = []
        actions = []
        
        for frame in tqdm(frames, desc="提取状态和动作", leave=False):
            # 提取状态
            state = self.state_extractor.extract(frame)
            states.append(state)
            
            # 提取动作（简化版本：随机选择动作）
            action_type = np.random.choice([
                'move', 'attack', 'skill_q', 'skill_w', 
                'skill_e', 'stop'
            ], p=[0.4, 0.3, 0.05, 0.05, 0.05, 0.15])
            
            action_data = {
                'type': action_type,
                'current_position': frame.get('hero_position', (0, 0)),
                'target_position': frame.get('target_position', 
                                              frame.get('hero_position', (0, 0)))
            }
            
            action = self.action_extractor.encode(action_data)
            actions.append(action)
        
        return np.array(states, dtype=np.float32), np.array(actions, dtype=np.int64)
    
    def _save_to_h5(self, output_path: str, states: np.ndarray, 
                    actions: np.ndarray, metadata: Dict[str, Any]):
        """保存为HDF5格式"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(output_path, 'w') as f:
            # 保存状态和动作
            f.create_dataset('states', data=states, compression='gzip')
            f.create_dataset('actions', data=actions, compression='gzip')
            
            # 保存元数据
            for key, value in metadata.items():
                if isinstance(value, (str, int, float)):
                    f.attrs[key] = value
            
            # 保存数据集信息
            f.attrs['num_samples'] = len(states)
            f.attrs['state_dim'] = states.shape[1] if len(states) > 0 else 0
            f.attrs['num_actions'] = len(np.unique(actions)) if len(actions) > 0 else 0
            f.attrs['conversion_time'] = str(Path.ctime(Path(output_path)))
    
    def convert_batch(self, replay_dir: str, output_dir: str, 
                    fps: int = 30, skip_frames: int = 4):
        """
        批量转换录像
        
        Args:
            replay_dir: 录像文件目录
            output_dir: 输出文件目录
            fps: 帧率
            skip_frames: 跳帧数
        """
        replay_path = Path(replay_dir)
        output_path = Path(output_dir)
        
        # 创建输出目录
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 获取所有.rofl文件
        replay_files = list(replay_path.glob('*.rofl'))
        
        print(f"找到 {len(replay_files)} 个.rofl文件")
        print(f"输出目录: {output_path}")
        
        # 转换每个文件
        for replay_file in tqdm(replay_files, desc="批量转换"):
            output_file = output_path / replay_file.with_suffix('.h5').name
            
            try:
                self.convert_single_replay(
                    str(replay_file),
                    str(output_file),
                    fps=fps,
                    skip_frames=skip_frames
                )
            except Exception as e:
                print(f"转换失败: {replay_file}, 错误: {e}")
                self.conversion_stats['failed'] += 1
                self.conversion_stats['failed_files'].append(str(replay_file))
        
        # 打印统计信息
        self._print_stats()
    
    def _print_stats(self):
        """打印转换统计"""
        print("\n" + "="*50)
        print("转换统计")
        print("="*50)
        print(f"总计: {self.conversion_stats['total']}")
        print(f"成功: {self.conversion_stats['success']}")
        print(f"失败: {self.conversion_stats['failed']}")
        
        if self.conversion_stats['total'] > 0:
            success_rate = self.conversion_stats['success'] / self.conversion_stats['total'] * 100
            print(f"成功率: {success_rate:.1f}%")
        
        if self.conversion_stats['failed_files']:
            print(f"\n失败的文件:")
            for failed_file in self.conversion_stats['failed_files']:
                print(f"  - {failed_file}")
        
        print("="*50 + "\n")
    
    def get_conversion_stats(self) -> Dict[str, Any]:
        """获取转换统计"""
        return self.conversion_stats
    
    def validate_output(self, output_path: str) -> bool:
        """验证输出文件是否有效
        
        Args:
            output_path: 输出文件路径
            
        Returns:
            is_valid: 是否有效
        """
        try:
            with h5py.File(output_path, 'r') as f:
                # 检查必要的dataset
                if 'states' not in f or 'actions' not in f:
                    return False
                
                # 检查数据形状
                states = f['states'][:]
                actions = f['actions'][:]
                
                if len(states) == 0 or len(actions) == 0:
                    return False
                
                if len(states) != len(actions):
                    return False
                
                print(f"验证通过: {output_path}")
                print(f"  - 样本数量: {len(states)}")
                print(f"  - 状态维度: {states.shape[1]}")
                print(f"  - 动作范围: {actions.min()} - {actions.max()}")
                
                return True
                
        except Exception as e:
            print(f"验证失败: {output_path}, 错误: {e}")
            return False

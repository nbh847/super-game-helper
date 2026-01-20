import struct
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import zstandard as zstd


class ReplayParser:
    """英雄联盟录像解析器
    
    解析.rofl文件并提取游戏数据
    """
    
    def __init__(self):
        self.replay_data = None
        self.metadata = {}
    
    def parse_replay(self, replay_path: str) -> Dict[str, Any]:
        """解析.rofl文件
        
        Args:
            replay_path: .rofl文件路径
            
        Returns:
            解析后的数据字典
        """
        try:
            replay_file = Path(replay_path)
            if not replay_file.exists():
                raise FileNotFoundError(f"录像文件不存在: {replay_path}")
            
            with open(replay_path, 'rb') as f:
                return self._parse_rofl_file(f)
                
        except Exception as e:
            print(f"解析录像失败: {e}")
            return None
    
    def _parse_rofl_file(self, f) -> Dict[str, Any]:
        """解析.rofl文件格式
        
        .rofl文件结构：
        - 文件头（magic number, version等）
        - 元数据（游戏信息）
        - 压缩的数据包
        """
        # 读取文件头
        header = self._read_header(f)
        
        # 读取元数据
        metadata = self._read_metadata(f)
        self.metadata = metadata
        
        # 读取压缩的数据
        payload_data = self._read_payload(f)
        
        # 解压数据
        decompressed_data = self._decompress_data(payload_data)
        
        # 解析数据包
        packets = self._parse_packets(decompressed_data)
        
        return {
            'header': header,
            'metadata': metadata,
            'packets': packets
        }
    
    def _read_header(self, f) -> Dict[str, Any]:
        """读取文件头"""
        # .rofl文件以特定magic number开头
        magic = f.read(4)
        if magic != b'RIOT':
            raise ValueError("无效的.rofl文件格式")
        
        # 读取版本号
        major_version = struct.unpack('<I', f.read(4))[0]
        minor_version = struct.unpack('<I', f.read(4))[0]
        
        return {
            'magic': magic,
            'major_version': major_version,
            'minor_version': minor_version
        }
    
    def _read_metadata(self, f) -> Dict[str, Any]:
        """读取元数据"""
        # 读取元数据长度
        metadata_length = struct.unpack('<I', f.read(4))[0]
        
        # 读取压缩的元数据
        compressed_metadata = f.read(metadata_length)
        
        # 解压元数据
        decompressed = zstd.decompress(compressed_metadata)
        
        # 解析JSON
        metadata_json = json.loads(decompressed.decode('utf-8'))
        
        return {
            'game_id': metadata_json.get('gameId', ''),
            'game_duration': metadata_json.get('gameDuration', 0),
            'game_mode': metadata_json.get('gameMode', ''),
            'players': metadata_json.get('players', []),
            'region': metadata_json.get('region', ''),
            'creation_time': metadata_json.get('creationTime', '')
        }
    
    def _read_payload(self, f) -> bytes:
        """读取压缩的数据负载"""
        # 读取payload长度
        payload_length = struct.unpack('<I', f.read(4))[0]
        
        # 读取payload数据
        return f.read(payload_length)
    
    def _decompress_data(self, compressed_data: bytes) -> bytes:
        """解压数据"""
        try:
            return zstd.decompress(compressed_data)
        except Exception as e:
            print(f"解压数据失败: {e}")
            return b''
    
    def _parse_packets(self, decompressed_data: bytes) -> List[Dict[str, Any]]:
        """解析数据包
        
        这里简化处理，实际需要根据LOL的协议格式解析
        """
        # 简化版本：返回空列表
        # 实际实现需要解析数据包格式
        packets = []
        
        # 这里可以添加数据包解析逻辑
        # 每个数据包包含时间戳、数据类型、数据内容等
        
        return packets
    
    def extract_frames(self, fps: int = 30) -> List[Dict[str, Any]]:
        """提取关键帧
        
        Args:
            fps: 帧率
            
        Returns:
            帧列表
        """
        if not self.replay_data:
            return []
        
        frames = []
        duration = self.metadata.get('game_duration', 0)
        num_frames = int(duration * fps)
        
        for i in range(num_frames):
            frame = {
                'timestamp': i / fps,
                'frame_number': i
            }
            frames.append(frame)
        
        return frames
    
    def extract_actions(self) -> List[Dict[str, Any]]:
        """提取玩家操作序列"""
        if not self.replay_data:
            return []
        
        actions = []
        packets = self.replay_data.get('packets', [])
        
        for packet in packets:
            if self._is_action_packet(packet):
                action = self._parse_action_packet(packet)
                if action:
                    actions.append(action)
        
        return actions
    
    def _is_action_packet(self, packet: Dict[str, Any]) -> bool:
        """判断是否是操作包"""
        # 简化版本：所有包都认为是操作包
        return True
    
    def _parse_action_packet(self, packet: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """解析操作包"""
        # 简化版本：返回基本动作信息
        return {
            'type': 'unknown',
            'timestamp': packet.get('timestamp', 0),
            'data': packet
        }
    
    def get_metadata(self, replay_path: str) -> Dict[str, Any]:
        """获取录像元数据"""
        data = self.parse_replay(replay_path)
        if data:
            return data.get('metadata', {})
        return {}
    
    def validate_replay(self, replay_path: str) -> bool:
        """验证录像文件是否有效"""
        try:
            data = self.parse_replay(replay_path)
            return data is not None
        except Exception as e:
            print(f"验证录像失败: {e}")
            return False

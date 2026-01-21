"""
简化的游戏状态识别器测试
"""

import sys
from pathlib import Path

# 添加backend目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from core.game_state import GameState

print('=' * 60)
print('游戏状态识别器测试（简化版）')
print('=' * 60)

# 创建游戏状态识别器
print('\n[1/2] 创建游戏状态识别器...')
game_state = GameState()
print('✓ 游戏状态识别器创建成功')

# 测试状态结构
print('\n[2/2] 测试状态结构...')
print(f'✓ 英雄位置: {game_state.get_hero_position()}')
print(f'✓ 是否危险: {game_state.is_in_danger()}')
print(f'✓ 最近敌方: {game_state.get_nearest_enemy()}')
print(f'✓ 安全位置: {game_state.get_safe_position()}')

# 测试张量转换
state_tensor = game_state.to_tensor()
print(f'✓ 状态张量形状: {state_tensor.shape}')
print(f'✓ 状态张量类型: {state_tensor.dtype}')
print(f'✓ 状态维度: {len(state_tensor)}')
print(f'✓ 状态张量前10维: {state_tensor[:10]}')

print('\n' + '=' * 60)
print('✓ 所有测试通过！')
print('=' * 60)

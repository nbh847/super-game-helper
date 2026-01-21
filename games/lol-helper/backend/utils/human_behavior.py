"""
人类行为模拟器（完善版）
模拟人类的随机性和不确定性，避免被检测为外挂
参考: docs/anti_detection.md
"""

import time
import random
import numpy as np
from typing import Optional, Tuple, List
from enum import Enum


class EmotionState(Enum):
    """情绪状态枚举"""
    NORMAL = "normal"
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    TIRED = "tired"
    EXCITED = "excited"


class GameContext(Enum):
    """游戏上下文枚举"""
    NORMAL = "normal"
    COMBAT = "combat"
    HIGH_STRESS = "high_stress"
    FARMING = "farming"
    ROAMING = "roaming"


class HumanBehaviorSimulator:
    """
    人类行为模拟器（完善版）
    
    添加随机延迟、不确定性，模拟人类操作
    参考: anti_detection.md 高级人类行为模拟
    """
    
    def __init__(self, 
                 reaction_time_range: Tuple[float, float] = (0.10, 0.18),
                 action_interval_range: Tuple[float, float] = (0.025, 0.08),
                 error_rate: float = 0.02):
        """
        初始化人类行为模拟器
        
        Args:
            reaction_time_range: 反应时间范围（秒），默认100-180ms（即时对战游戏）
            action_interval_range: 动作间隔范围（秒），默认25-80ms（对应200-400 APM）
            error_rate: 错误率（0.0-1.0），默认2%
        """
        # 基础参数
        self.reaction_time_range = reaction_time_range
        self.action_interval_range = action_interval_range
        self.error_rate = error_rate
        
        # 玩家画像（随机生成）
        self.profile = self._generate_profile()
        
        # 状态变量
        self.current_emotion = EmotionState.NORMAL
        self.current_context = GameContext.NORMAL
        self.fatigue = 0.0  # 0.0-1.0，疲劳度
        self.kill_streak = 0
        self.death_streak = 0
        
        # 统计
        self.last_action_time = 0
        self.action_count = 0
        self.error_count = 0
        self.session_start_time = time.time()
        
        print(f"[HumanBehavior] 人类行为模拟器初始化成功")
        print(f"[HumanBehavior] 玩家画像: APM={self.profile['base_apm']}, "
              f"反应={self.profile['base_reaction']*1000:.0f}ms, "
              f"激进度={self.profile['aggression']:.2f}")
    
    def _generate_profile(self) -> dict:
        """
        生成个性化玩家画像
        
        Returns:
            玩家画像字典
        """
        return {
            "base_apm": random.randint(200, 400),           # APM 200-400（即时对战游戏）
            "base_reaction": random.uniform(0.10, 0.18),  # 反应时间 100-180ms
            "aggression": random.uniform(0.4, 0.8),         # 激进度 0.4-0.8（偏激进）
            "focus": random.uniform(0.75, 0.98),            # 专注度 0.75-0.98
            "stability": random.uniform(0.65, 0.92),          # 稳定性 0.65-0.92
        }
    
    def get_reaction_time(self) -> float:
        """
        获取动态反应时间（受疲劳、情绪、上下文影响）
        即时对战游戏优化：保持100-200ms范围
        
        Returns:
            反应时间（秒）
        """
        base = self.profile['base_reaction']  # 100-180ms
        
        # 疲劳影响（疲劳度每增加0.1，反应时间增加15%，最多250ms）
        fatigue_factor = 1.0 + min(self.fatigue * 1.5, 0.4)
        
        # 情绪影响（即时对战游戏中，情绪对反应影响较小）
        emotion_factor = 1.0
        if self.current_emotion == EmotionState.EXCITED:
            emotion_factor = 0.85  # 激动时快15%
        elif self.current_emotion == EmotionState.TIRED:
            emotion_factor = 1.2  # 疲劳时慢20%（不是30%）
        
        # 上下文影响（战斗时反应更快）
        context_factor = 1.0
        if self.current_context == GameContext.HIGH_STRESS:
            context_factor = 0.9  # 高压时快10%（不是30%）
        elif self.current_context == GameContext.COMBAT:
            context_factor = 0.95  # 战斗时快5%
        elif self.current_context == GameContext.FARMING:
            context_factor = 1.1  # 补刀时慢10%（不是20%）
        
        # 专注度影响
        focus_factor = 1.0 / self.profile['focus']
        
        # 自然随机性（±10%）
        random_factor = random.uniform(0.9, 1.1)
        
        reaction_time = (base * fatigue_factor * emotion_factor * 
                        context_factor * focus_factor * random_factor)
        
        # 限制在80-250ms范围内（既不要太快被检测，也不能太慢无法游戏）
        reaction_time = max(0.08, min(0.25, reaction_time))
        
        # 注意力分散（5%概率额外延迟，但不超过50ms）
        if random.random() < 0.05:
            extra_delay = random.uniform(0.01, 0.05)
            reaction_time += extra_delay
        
        return reaction_time
    
    def add_delay(self, custom_delay: Optional[float] = None) -> None:
        """
        添加延迟
        
        Args:
            custom_delay: 自定义延迟，None表示使用动态反应时间
        """
        if custom_delay is None:
            delay = self.get_reaction_time()
        else:
            delay = custom_delay
        
        time.sleep(delay)
        self.last_action_time = time.time()
        self.action_count += 1
    
    def get_move_speed(self, base_speed: Optional[float] = None) -> float:
        """
        获取移动速度（添加随机性）
        
        Args:
            base_speed: 基础速度，None表示自动计算
            
        Returns:
            移动速度（秒）
        """
        if base_speed is None:
            base_speed = 1.0
        
        # 添加随机性（±5%）
        random_factor = random.uniform(0.95, 1.05)
        
        # 情绪影响（即时对战游戏中，对移动速度影响较小）
        emotion_factor = 1.0
        if self.current_emotion == EmotionState.AGGRESSIVE:
            emotion_factor = 0.98  # 激进时稍快2%
        elif self.current_emotion == EmotionState.TIRED:
            emotion_factor = 1.03  # 疲劳时稍慢3%
        
        speed = base_speed * random_factor * emotion_factor
        return speed
    
    def get_action_interval(self) -> float:
        """
        获取随机动作间隔（受玩家画像和状态影响）
        即时对战游戏优化：保持150-500ms间隔（对应200-400 APM）
        
        Returns:
            动作间隔（秒）
        """
        # APM转换为秒（例如：200 APM = 60/200 = 0.3秒间隔）
        base_interval = 60.0 / self.profile['base_apm']  # 150-375ms
        
        # 情绪影响（即时对战游戏中，对APM影响较小）
        if self.current_emotion == EmotionState.AGGRESSIVE:
            base_interval *= 0.92  # 激进时快8%（不是10%）
        elif self.current_emotion == EmotionState.CONSERVATIVE:
            base_interval *= 1.15  # 保守时慢15%（不是20%）
        
        # 上下文影响（战斗时操作更频繁）
        if self.current_context == GameContext.COMBAT:
            base_interval *= 0.85  # 战斗时快15%
        elif self.current_context == GameContext.HIGH_STRESS:
            base_interval *= 0.8  # 高压时快20%
        elif self.current_context == GameContext.FARMING:
            base_interval *= 1.1  # 补刀时慢10%
        
        # 疲劳影响（操作越多，间隔略微增大）
        if self.action_count > 200:
            fatigue_factor = 1.0 + (self.action_count / 10000.0)  # 每1000个操作增加10%
            base_interval *= min(fatigue_factor, 1.2)  # 最多增加20%
        
        # 随机性（±15%）
        random_factor = random.uniform(0.85, 1.15)
        
        interval = base_interval * random_factor
        
        # 限制在100-500ms范围内（对应120-600 APM）
        # 既不会因为太慢被"不会玩"，也不会因为太快被检测
        interval = max(0.10, min(0.50, interval))
        
        return interval
    
    def generate_mouse_trajectory(self, start_pos: Tuple[int, int], 
                              end_pos: Tuple[int, int],
                              movement_type: str = "normal") -> List[Tuple[int, int]]:
        """
        生成自然鼠标轨迹（Catmull-Rom + Perlin Noise）
        
        Args:
            start_pos: 起始位置 (x, y)
            end_pos: 目标位置 (x, y)
            movement_type: 移动类型（normal/flick/precise）
            
        Returns:
            轨迹点列表
        """
        if movement_type == "flick":
            points = self._flick_trajectory(start_pos, end_pos)
        elif movement_type == "precise":
            points = self._precise_trajectory(start_pos, end_pos)
        else:
            points = self._normal_trajectory(start_pos, end_pos)
        
        return points
    
    def _normal_trajectory(self, start_pos: Tuple[int, int], 
                        end_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        正常移动轨迹（平滑曲线 + 速度变化）
        """
        # 控制点（Catmull-Rom样条需要至少4个点）
        points = [start_pos]
        
        # 添加中间控制点（贝塞尔曲线效果）
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        
        # 2个控制点，创建曲线
        cp1 = (int(start_pos[0] + dx * 0.25 + random.randint(-20, 20)),
                int(start_pos[1] + dy * 0.25 + random.randint(-20, 20)))
        cp2 = (int(start_pos[0] + dx * 0.75 + random.randint(-20, 20)),
                int(start_pos[1] + dy * 0.75 + random.randint(-20, 20)))
        
        points.extend([cp1, cp2, end_pos])
        
        # 生成轨迹点（12个点）
        trajectory = self._catmull_rom_spline(points, num_points=12)
        
        # 添加Perlin Noise
        trajectory = self._add_perlin_noise(trajectory, intensity=2.0)
        
        # 应用速度曲线（加速→匀速→减速）
        trajectory = self._apply_velocity_profile(trajectory)
        
        return trajectory
    
    def _flick_trajectory(self, start_pos: Tuple[int, int], 
                         end_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        快速甩动轨迹（如瞄准）
        """
        points = [start_pos]
        
        # 控制点（直线）
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        mid = (int(start_pos[0] + dx * 0.5),
                int(start_pos[1] + dy * 0.5))
        
        points.extend([mid, end_pos])
        
        # 生成轨迹点（5个点，快速）
        trajectory = self._catmull_rom_spline(points, num_points=5)
        
        # 添加轻微Perlin Noise
        trajectory = self._add_perlin_noise(trajectory, intensity=1.0)
        
        return trajectory
    
    def _precise_trajectory(self, start_pos: Tuple[int, int], 
                           end_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        精细操作轨迹（如点击小目标）
        """
        points = [start_pos]
        
        # 多个控制点（高精度）
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        
        for t in [0.25, 0.5, 0.75]:
            cp = (int(start_pos[0] + dx * t + random.randint(-10, 10)),
                  int(start_pos[1] + dy * t + random.randint(-10, 10)))
            points.append(cp)
        
        points.append(end_pos)
        
        # 生成轨迹点（15个点）
        trajectory = self._catmull_rom_spline(points, num_points=15)
        
        # 添加中等Perlin Noise
        trajectory = self._add_perlin_noise(trajectory, intensity=1.5)
        
        # 在目标附近微调（最后2个点）
        if len(trajectory) >= 2:
            last = trajectory[-1]
            second_last = trajectory[-2]
            trajectory[-1] = (
                int(last[0] + random.randint(-2, 2)),
                int(last[1] + random.randint(-2, 2))
            )
            trajectory[-2] = (
                int(second_last[0] + random.randint(-1, 1)),
                int(second_last[1] + random.randint(-1, 1))
            )
        
        return trajectory
    
    def _catmull_rom_spline(self, points: List[Tuple[int, int]], 
                            num_points: int = 10) -> List[Tuple[int, int]]:
        """
        Catmull-Rom样条曲线生成
        
        Args:
            points: 控制点列表（至少4个点）
            num_points: 生成的轨迹点数
            
        Returns:
            轨迹点列表
        """
        if len(points) < 4:
            # 控制点不足，使用线性插值
            trajectory = []
            for i in range(num_points):
                t = i / (num_points - 1)
                x = int(points[0][0] + (points[-1][0] - points[0][0]) * t)
                y = int(points[0][1] + (points[-1][1] - points[0][1]) * t)
                trajectory.append((x, y))
            return trajectory
        
        trajectory = []
        
        # Catmull-Rom参数化
        def catmull_rom_spline(p0, p1, p2, p3, t):
            """
            Catmull-Rom样条函数
            
            Args:
                p0, p1, p2, p3: 四个控制点
                t: 参数（0到1之间）
                
            Returns:
                插值点
            """
            t2 = t * t
            t3 = t2 * t
            
            # Catmull-Rom基函数
            f0 = -0.5 * t3 + t2 - 0.5 * t
            f1 = 1.5 * t3 - 2.5 * t2 + 1.0
            f2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
            f3 = 0.5 * t3 - 0.5 * t2
            
            x = f0 * p0[0] + f1 * p1[0] + f2 * p2[0] + f3 * p3[0]
            y = f0 * p0[1] + f1 * p1[1] + f2 * p2[1] + f3 * p3[1]
            
            return (int(x), int(y))
        
        # 生成轨迹点
        for i in range(num_points):
            t = i / (num_points - 1)
            
            # 找到对应的四个控制点
            segment = int(t * (len(points) - 3))
            segment = max(0, min(segment, len(points) - 4))
            
            p0 = points[segment]
            p1 = points[segment + 1]
            p2 = points[segment + 2]
            p3 = points[segment + 3]
            
            # 局部参数
            local_t = (t * (len(points) - 3)) - segment
            
            # 插值
            point = catmull_rom_spline(p0, p1, p2, p3, local_t)
            trajectory.append(point)
        
        return trajectory
    
    def _add_perlin_noise(self, points: List[Tuple[int, int]], 
                       intensity: float = 1.0) -> List[Tuple[int, int]]:
        """
        添加Perlin噪声（简化版本，使用1/f噪声）
        
        Args:
            points: 轨迹点列表
            intensity: 噪声强度
            
        Returns:
            添加噪声后的轨迹点
        """
        if len(points) < 2:
            return points
        
        noisy_points = []
        
        # 生成1/f噪声（低频成分更强）
        num_points = len(points)
        
        # 低频噪声（全局弯曲）
        low_freq = int(random.gauss(0, 1) * intensity * 10)
        
        for i, (x, y) in enumerate(points):
            # 1/f噪声：远距离点相关性强
            freq = (i / num_points) * 2 * np.pi
            
            noise_x = int(np.sin(freq) * intensity + low_freq)
            noise_y = int(np.cos(freq) * intensity + low_freq)
            
            # 添加噪声
            noisy_x = x + noise_x
            noisy_y = y + noise_y
            
            noisy_points.append((noisy_x, noisy_y))
        
        return noisy_points
    
    def _apply_velocity_profile(self, points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        应用速度曲线（加速→匀速→减速）
        
        Args:
            points: 轨迹点列表
            
        Returns:
            应用速度曲线后的轨迹点
        """
        if len(points) < 3:
            return points
        
        # 简化实现：调整点的密度（中间稀疏，两端密集）
        # 这里不做实际改变，只是记录速度曲线的概念
        
        return points
    
    def add_jitter(self, position: tuple, 
                  jitter_range: int = 5) -> tuple:
        """
        添加位置抖动（模拟手部微动）
        
        Args:
            position: 原始位置 (x, y)
            jitter_range: 抖动范围（像素）
            
        Returns:
            抖动后的位置
        """
        x, y = position
        jitter_x = random.randint(-jitter_range, jitter_range)
        jitter_y = random.randint(-jitter_range, jitter_range)
        
        return (x + jitter_x, y + jitter_y)
    
    def calculate_mistake_probability(self) -> float:
        """
        计算失误概率（受疲劳、情绪、上下文影响）
        
        Returns:
            失误概率（0.0-1.0）
        """
        # 基础失误率
        base_error = self.error_rate  # 默认2%
        
        # 疲劳因素（每增加0.1疲劳度，失误率增加0.8%）
        fatigue_factor = self.fatigue * 0.08
        
        # 情绪因素
        emotion_factor = 0.0
        if self.current_emotion == EmotionState.TIRED:
            emotion_factor = 0.05  # 疲劳时失误率+5%
        elif self.current_emotion == EmotionState.EXCITED:
            emotion_factor = 0.03  # 激动时失误率+3%
        elif self.current_emotion == EmotionState.CONSERVATIVE:
            emotion_factor = -0.02  # 保守时失误率-2%
        
        # 上下文因素
        context_factor = 0.0
        if self.current_context == GameContext.HIGH_STRESS:
            context_factor = 0.02  # 高压时失误率+2%
        elif self.current_context == GameContext.FARMING:
            context_factor = -0.01  # 补刀时失误率-1%
        
        # 玩家稳定性因素
        stability_factor = (1.0 - self.profile['stability']) * 0.03
        
        # 专注度因素
        focus_factor = (1.0 - self.profile['focus']) * 0.02
        
        # 总失误率
        total_error = (base_error + fatigue_factor + emotion_factor + 
                      context_factor + stability_factor + focus_factor)
        
        return min(0.15, max(0.01, total_error))  # 限制在1%-15%
    
    def should_make_error(self) -> bool:
        """
        判断是否应该犯错
        
        Returns:
            是否犯错
        """
        return random.random() < self.calculate_mistake_probability()
    
    def get_random_error_type(self) -> str:
        """
        获取随机错误类型
        
        Returns:
            错误类型
        """
        error_types = [
            'miss_click',      # 点击偏移
            'wrong_target',    # 选错目标
            'mistimed_skill',  # 技能时机不对
            'wrong_direction',  # 移动方向错误
            'cancel_action'     # 取消操作
        ]
        return random.choice(error_types)
    
    def get_humanized_position(self, target_pos: tuple, 
                              current_pos: Optional[tuple] = None) -> tuple:
        """
        获取人类化的目标位置（包含路径偏移）
        
        Args:
            target_pos: 目标位置 (x, y)
            current_pos: 当前位置 (x, y)
            
        Returns:
            人类化的位置
        """
        # 添加抖动
        humanized_pos = self.add_jitter(target_pos)
        
        # 如果有当前位置，添加路径偏移（模拟走位）
        if current_pos is not None:
            dx = target_pos[0] - current_pos[0]
            dy = target_pos[1] - current_pos[1]
            distance = (dx**2 + dy**2)**0.5
            
            if distance > 0:
                # 路径偏移（模拟走位，±10像素）
                offset = random.uniform(-10, 10)
                humanized_pos = (
                    int(target_pos[0] + offset),
                    int(target_pos[1] + offset)
                )
        
        return humanized_pos
    
    def update_state(self, event_type: str, 
                   event_data: Optional[dict] = None) -> None:
        """
        根据游戏事件更新状态
        
        Args:
            event_type: 事件类型（kill/death/assist/session_time/context）
            event_data: 事件数据
        """
        if event_type == "kill":
            self.kill_streak += 1
            self.death_streak = 0
            if self.kill_streak >= 3:
                self.current_emotion = EmotionState.EXCITED
                print(f"[HumanBehavior] 击杀连杀{self.kill_streak}，情绪：激动")
        
        elif event_type == "death":
            self.death_streak += 1
            self.kill_streak = 0
            if self.death_streak >= 2:
                self.current_emotion = EmotionState.CONSERVATIVE
                print(f"[HumanBehavior] 死亡{self.death_streak}次，情绪：保守")
        
        elif event_type == "session_time":
            # 更新疲劳度
            session_minutes = event_data.get('minutes', 0) if event_data else 0
            self.fatigue = min(1.0, session_minutes / 240.0)  # 每4小时疲劳度+1
            if self.fatigue > 0.7:
                self.current_emotion = EmotionState.TIRED
                print(f"[HumanBehavior] 疲劳度: {self.fatigue:.2f}，情绪：疲劳")
        
        elif event_type == "context":
            # 更新上下文
            if event_data:
                context_str = event_data.get('context', 'normal')
                if context_str == 'combat':
                    self.current_context = GameContext.COMBAT
                elif context_str == 'high_stress':
                    self.current_context = GameContext.HIGH_STRESS
                elif context_str == 'farming':
                    self.current_context = GameContext.FARMING
                elif context_str == 'roaming':
                    self.current_context = GameContext.ROAMING
                else:
                    self.current_context = GameContext.NORMAL
    
    def simulate_fatigue(self, action_count: int) -> bool:
        """
        模拟疲劳（长时间操作后性能下降）
        
        Args:
            action_count: 已执行的操作数量
            
        Returns:
            是否疲劳
        """
        # 每2000个操作检查一次
        if action_count > 0 and action_count % 2000 == 0:
            # 随着操作增加，疲劳概率上升（最多20%概率）
            fatigue_prob = min(action_count / 20000.0, 0.2)
            return random.random() < fatigue_prob
        
        return False
    
    def get_break_time(self) -> Optional[float]:
        """
        获取休息时间（模拟人类需要休息）
        
        Returns:
            休息时间（秒），None表示不休息
        """
        # 每500个操作有5%概率休息
        if self.action_count > 0 and self.action_count % 500 == 0:
            if random.random() < 0.05:
                # 休息时长（短5-10分钟，中10-20分钟，长20-30分钟）
                break_type = random.choice(['short', 'medium', 'long'])
                
                if break_type == 'short':
                    break_time = random.uniform(5 * 60, 10 * 60)  # 5-10分钟
                elif break_type == 'medium':
                    break_time = random.uniform(10 * 60, 20 * 60)  # 10-20分钟
                else:
                    break_time = random.uniform(20 * 60, 30 * 60)  # 20-30分钟
                
                print(f"[HumanBehavior] 需要休息，类型: {break_type}, 时长: {break_time/60:.1f}分钟")
                return break_time
        
        return None
    
    def reset_stats(self) -> None:
        """重置统计数据"""
        self.last_action_time = 0
        self.action_count = 0
        self.error_count = 0
        self.session_start_time = time.time()
        self.fatigue = 0.0
        self.current_emotion = EmotionState.NORMAL
        print("[HumanBehavior] 统计数据已重置")
    
    def get_stats(self) -> dict:
        """
        获取统计数据
        
        Returns:
            统计数据字典
        """
        session_duration = time.time() - self.session_start_time
        
        return {
            "action_count": self.action_count,
            "error_count": self.error_count,
            "error_rate": (self.error_count / self.action_count 
                          if self.action_count > 0 else 0),
            "session_duration": session_duration,  # 秒
            "fatigue": self.fatigue,
            "current_emotion": self.current_emotion.value,
            "current_context": self.current_context.value,
            "profile": self.profile
        }
    
    def benchmark(self, iterations: int = 100) -> dict:
        """
        性能测试
        
        Args:
            iterations: 测试次数
            
        Returns:
            性能统计字典
        """
        print(f"[HumanBehavior] 开始性能测试，次数: {iterations}")
        
        reaction_times = []
        intervals = []
        
        for i in range(iterations):
            # 测试反应时间
            start = time.time()
            self.add_delay()
            reaction_times.append(time.time() - start)
            
            # 测试动作间隔
            interval = self.get_action_interval()
            intervals.append(interval)
            
            # 测试轨迹生成
            if i % 10 == 0:
                trajectory = self.generate_mouse_trajectory((400, 400), (600, 600))
        
        # 计算统计数据
        result = {
            "iterations": iterations,
            "reaction_time": {
                "mean": np.mean(reaction_times) * 1000,  # 毫秒
                "min": np.min(reaction_times) * 1000,
                "max": np.max(reaction_times) * 1000,
                "std": np.std(reaction_times) * 1000
            },
            "action_interval": {
                "mean": np.mean(intervals) * 1000,  # 毫秒
                "min": np.min(intervals) * 1000,
                "max": np.max(intervals) * 1000,
                "std": np.std(intervals) * 1000
            },
            "error_rate": self.calculate_mistake_probability()
        }
        
        print(f"[HumanBehavior] 性能测试完成:")
        print(f"  反应时间: 平均{result['reaction_time']['mean']:.2f}ms, "
              f"范围{result['reaction_time']['min']:.2f}-{result['reaction_time']['max']:.2f}ms")
        print(f"  动作间隔: 平均{result['action_interval']['mean']:.2f}ms, "
              f"范围{result['action_interval']['min']:.2f}-{result['action_interval']['max']:.2f}ms")
        print(f"  预计失误率: {result['error_rate']*100:.2f}%")
        
        return result


# 测试代码
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    print("=" * 60)
    print("人类行为模拟器测试（完善版）")
    print("=" * 60)
    
    # 创建模拟器
    print("\n[1/6] 创建人类行为模拟器...")
    simulator = HumanBehaviorSimulator()
    print("✓ 人类行为模拟器创建成功")
    
    # 测试动态反应时间
    print("\n[2/6] 测试动态反应时间...")
    print("  测试不同情绪和疲劳下的反应时间：")
    simulator.current_emotion = EmotionState.NORMAL
    simulator.fatigue = 0.0
    print(f"  - 正常: {simulator.get_reaction_time()*1000:.1f}ms")
    
    simulator.current_emotion = EmotionState.EXCITED
    print(f"  - 激动: {simulator.get_reaction_time()*1000:.1f}ms")
    
    simulator.current_emotion = EmotionState.TIRED
    simulator.fatigue = 0.5
    print(f"  - 疲劳: {simulator.get_reaction_time()*1000:.1f}ms")
    
    simulator.current_emotion = EmotionState.CONSERVATIVE
    simulator.current_context = GameContext.COMBAT
    print(f"  - 战斗+保守: {simulator.get_reaction_time()*1000:.1f}ms")
    
    # 测试轨迹生成
    print("\n[3/6] 测试轨迹生成...")
    trajectory = simulator.generate_mouse_trajectory((400, 400), (600, 600))
    print(f"  ✓ 轨迹点数: {len(trajectory)}")
    print(f"  ✓ 起始点: {trajectory[0]}")
    print(f"  ✓ 终点: {trajectory[-1]}")
    
    # 测试失误概率
    print("\n[4/6] 测试失误概率...")
    print(f"  - 正常: {simulator.calculate_mistake_probability()*100:.2f}%")
    simulator.current_emotion = EmotionState.TIRED
    simulator.fatigue = 0.8
    print(f"  - 疲劳: {simulator.calculate_mistake_probability()*100:.2f}%")
    
    # 测试状态更新
    print("\n[5/6] 测试状态更新...")
    simulator.update_state("kill")
    print(f"  ✓ 情绪: {simulator.current_emotion.value}")
    
    simulator.update_state("death")
    print(f"  ✓ 情绪: {simulator.current_emotion.value}")
    
    simulator.update_state("context", {"context": "combat"})
    print(f"  ✓ 上下文: {simulator.current_context.value}")
    
    # 性能测试
    print("\n[6/6] 性能测试（100次）...")
    result = simulator.benchmark(iterations=100)
    
    # 统计数据
    print("\n[7/7] 统计数据...")
    stats = simulator.get_stats()
    print(f"  ✓ 玩家画像: APM={stats['profile']['base_apm']}, "
          f"反应={stats['profile']['base_reaction']*1000:.0f}ms")
    print(f"  ✓ 总操作数: {stats['action_count']}")
    print(f"  ✓ 当前情绪: {stats['current_emotion']}")
    print(f"  ✓ 当前上下文: {stats['current_context']}")
    print(f"  ✓ 疲劳度: {stats['fatigue']:.2f}")
    
    print("\n" + "=" * 60)
    print("✓ 所有测试完成！")
    print("=" * 60)

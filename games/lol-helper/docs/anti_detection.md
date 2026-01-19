# 防检测策略

## 外挂 vs AI 的本质区别

### 🎮 外挂（作弊器）

```
直接修改游戏数据或内存：
- 锁血（血量不变）
- 自动秒人（伤害修改）
- 透视（显示隐藏信息）
- 自动躲避（修改碰撞判定）

检测特征：
- ✅ 内存修改痕迹明显
- ✅ 不自然的完美操作（0ms反应）
- ✅ 违背游戏规则的数据
```
**结果：秒封**

### 🤖 AI（智能体）

```
通过屏幕识别 + 键鼠操作：
- 看到屏幕 → 决策 → 操作（和人类一样）
- 不修改游戏任何数据
- 通过正常输入设备操作

检测特征：
- ⚠️ 可能只有异常的数据表现
- ⚠️ 非人类的作息模式
- ⚠️ 过于完美的操作
```
**结果：更难检测**

## Grok AI 案例分析与启示

### Grok AI 的实际表现

- 92% 胜率（56 场 52 胜 4 负）
- 精通 22 个英雄
- 每天 12:00-02:00 固定时段上线
- 连续 14 小时无休
- 远超人类顶尖选手水平

### 为什么 Grok AI 暂时未被封

#### 技术层面的"合法性"

```
✅ 不修改游戏内存
✅ 不读写游戏进程
✅ 通过正常的键盘鼠标输入
✅ 操作在人类可达范围内（虽然很快）

反作弊系统检测的是：
❌ 内存注入/修改
❌ 游戏数据篡改
❌ 异常进程交互
❌ Hook API 调用
```

#### 可能的检测时机

- 拳头公司可能正在收集数据
- 需要开发 AI 检测 AI 的技术
- 但明显特征迟早会被识别

### Grok AI 的明显异常特征（应避免）

```
🔴 作息过于规律：
   - 每天12:00准时上线
   - 02:00后下线
   - 时间标准差几乎为0

🔴 超人类耐力：
   - 连续14小时无休
   - 中间只停10-30分钟
   - 远超人类生理极限

🔴 异常胜率：
   - 92%胜率
   - 远超人类顶尖选手的60-70%
   - 过于完美的数据

🔴 英雄广度：
   - 精通22个英雄
   - 人类通常专注1-3个英雄
```

### Grok AI 对我们的启示

#### 应该学习的
1. 技术路线验证（行为克隆 + 强化学习）
2. 多英雄泛化能力
3. 高质量训练数据的重要性

#### 应该避免的
1. 过于规律的作息模式
2. 异常的高胜率
3. 超人类的持续操作时间
4. 过于完美的表现

## 拳头的反作弊机制

### 传统反作弊（容易检测）

```python
# 内存扫描
if detect_memory_modification():
    ban_account()

# 异常数据
if health_locked or damage_modified:
    ban_account()

# 完美操作
if reaction_time < 50ms:  # 人类极限约100ms
    flag_account()
```

### 行为分析反作弊（中等难度）

```python
# 统计特征分析
if win_rate > 0.8 and hours_per_day > 12:
    suspect_AI()

# 操作模式分析
if APM < 100 and win_rate > 0.8:
    # 低操作量但高胜率，可疑
    flag_account()

# 作息模式分析
if login_time_distribution.std() < 0.5:  # 太规律
    suspect_bot()
```

### AI 检测 AI（未来趋势）

```python
# 训练一个分类器
classifier = TrainOnHumanVsAIData()

# 判断操作是否来自AI
if classifier.predict(actions) == "AI":
    ban_account()
```

## 我们的防检测策略

### 技术层面

#### 1. 完全"合法"的操作方式

```python
# 只用屏幕识别 + 键鼠输入
screen = capture_screen()
decision = AI_model(screen)
execute_action(decision)  # 正常的 pyautogui 操作

# 不做任何游戏数据修改
# ✅ 符合反作弊规则
```

#### 2. 人类可达的操作范围

```python
class HumanLikeAction:
    # 反应时间：150-300ms（人类范围）
    def reaction_time(self):
        return random.randint(150, 300)

    # APM：150-250（正常玩家）
    def apm_control(self):
        return random.randint(150, 250)

    # 准确率：85-95%（会失误）
    def accuracy(self):
        return random.uniform(0.85, 0.95)

    # 操作速度：在人类范围内
    def move_speed(self, distance):
        # 人类鼠标速度：750-880 像素/秒
        speed = random.randint(750, 880)
        return distance / speed
```

#### 3. 不修改任何游戏数据

```python
# ✅ 正确（我们的方式）
screen = capture_game_window()  # 只读取屏幕
decision = ai_infer(screen)     # AI 决策
input_simulator.send(decision)  # 模拟键鼠

# ❌ 错误（外挂方式）
modify_memory(address, value)    # 修改内存
hook_game_function(func)         # Hook 函数
read_process_memory(pid)         # 读取进程内存
```

### 行为层面

#### 1. 模拟人类作息

```python
class HumanSchedule:
    def __init__(self):
        # 多个合理的时段
        self.time_slots = [
            ("10:00-14:00", 4),
            ("15:00-19:00", 4),
            ("20:00-00:00", 4),
            ("21:00-02:00", 5)
        ]

    def get_daily_schedule(self):
        """随机选择一天的时段"""
        num_slots = random.randint(1, 2)
        slots = random.sample(self.time_slots, num_slots)

        # 添加随机休息日
        if random.random() < 0.2:  # 20%概率休息
            return []

        return slots

    def play_duration_per_session(self):
        """每次上线的时长（小时）"""
        return random.uniform(2, 4)  # 2-4小时
```

#### 2. 定期休息机制

```python
class BreakSchedule:
    def __init__(self):
        self.session_games = 0
        self.total_play_time = 0

    def should_take_break(self):
        """判断是否应该休息"""
        # 每3-5局休息
        if self.session_games >= random.randint(3, 5):
            return True

        # 每1-2小时休息
        if self.total_play_time >= random.randint(3600, 7200):
            return True

        return False

    def break_duration(self):
        """休息时长"""
        break_type = random.choice([
            ("short", random.uniform(5, 10)),    # 短休息
            ("medium", random.uniform(10, 20)),  # 中等休息
            ("long", random.uniform(20, 30))     # 长休息
        ])
        return break_type[1]
```

#### 3. 胜率控制

```python
class WinRateControl:
    def __init__(self, target_win_rate=0.55):
        self.target_win_rate = target_win_rate  # 目标胜率55%
        self.current_win_rate = 0.0

    def should_intentionally_lose(self, current_stats):
        """判断是否应该故意输一局"""
        wins, total = current_stats
        self.current_win_rate = wins / total

        # 胜率超过65%，故意输一局
        if self.current_win_rate > 0.65:
            return True

        return False

    def intentional_mistake_level(self):
        """失误程度"""
        return random.choice([
            "miss_skill",       # 空技能
            "misposition",      # 走位失误
            "overextend",       # 走位过深
            "miss_timing"       # 时机不对
        ])
```

#### 4. 高级人类行为模拟

基于成熟的人机行为模拟方法，包括：
- Catmull-Rom 样条曲线生成自然鼠标轨迹
- Perlin Noise 添加自然随机性
- 疲劳模型：随时间推移增加反应时间和失误率
- 情绪模型：激进/保守/疲劳/兴奋状态
- 上下文感知：根据游戏状态调整行为
- 微动作：模拟真实玩家的微小抖动和预判

```python
class AdvancedHumanBehavior:
    """高级人类行为模拟器"""

    def __init__(self):
        # 玩家画像
        self.profile = self._generate_profile()

        # 状态变量
        self.fatigue = 0.0
        self.current_emotion = "normal"
        self.kill_streak = 0
        self.death_streak = 0

    def _generate_profile(self):
        """生成个性化玩家画像"""
        return {
            "base_apm": random.randint(150, 280),
            "base_reaction": random.uniform(0.15, 0.3),
            "aggression": random.uniform(0.3, 0.7),
            "focus": random.uniform(0.7, 0.95)
        }

    def get_dynamic_reaction_time(self, context="normal"):
        """
        获取动态反应时间（受疲劳、情绪、上下文影响）

        Args:
            context: 上下文（normal/combat/high_stress）

        Returns:
            reaction_time: 反应时间（秒）
        """
        base = self.profile["base_reaction"]

        # 疲劳影响
        fatigue_factor = 1.0 + (self.fatigue * 0.5)

        # 情绪影响
        emotion_factor = 1.0
        if self.current_emotion == "excited":
            emotion_factor = 0.8
        elif self.current_emotion == "tired":
            emotion_factor = 1.5

        # 上下文影响
        context_factor = 1.0
        if context == "high_stress":
            context_factor = 0.6

        # 自然随机性
        random_factor = random.uniform(0.9, 1.1)

        return base * fatigue_factor * emotion_factor * context_factor * random_factor

    def generate_mouse_trajectory(self, start_pos, end_pos, movement_type="normal"):
        """
        生成自然鼠标轨迹

        Args:
            start_pos: 起始位置
            end_pos: 目标位置
            movement_type: 移动类型（normal/flick/precise）

        Returns:
            points: 轨迹点列表
        """
        if movement_type == "flick":
            # 快速甩动（如瞄准）
            return self._flick_trajectory(start_pos, end_pos)
        elif movement_type == "precise":
            # 精细操作（如点击小目标）
            return self._precise_trajectory(start_pos, end_pos)
        else:
            # 正常移动
            return self._normal_trajectory(start_pos, end_pos)

    def _normal_trajectory(self, start_pos, end_pos):
        """正常移动轨迹（Catmull-Rom + Perlin Noise）"""
        # Catmull-Rom 样条曲线
        points = self._catmull_rom_spline([start_pos, end_pos], num_points=12)

        # Perlin Noise
        points = self._add_perlin_noise(points, intensity=2.0)

        # 速度曲线（加速 -> 匀速 -> 减速）
        points = self._apply_velocity_profile(points)

        return points

    def _flick_trajectory(self, start_pos, end_pos):
        """快速甩动轨迹"""
        points = self._catmull_rom_spline([start_pos, end_pos], num_points=5)
        points = self._add_perlin_noise(points, intensity=1.0)
        return points

    def _precise_trajectory(self, start_pos, end_pos):
        """精细操作轨迹"""
        points = self._catmull_rom_spline([start_pos, end_pos], num_points=15)
        points = self._add_perlin_noise(points, intensity=1.5)

        # 在目标附近微调
        last = points[-1]
        points[-1] = (last[0] + random.randint(-2, 2), last[1] + random.randint(-2, 2))

        return points

    def _catmull_rom_spline(self, points, num_points):
        """Catmull-Rom 样条曲线生成"""
        # 实现略...（见 API 参考）
        pass

    def _add_perlin_noise(self, points, intensity):
        """添加 Perlin Noise"""
        # 实现略...（见 API 参考）
        pass

    def _apply_velocity_profile(self, points):
        """应用速度曲线"""
        # 实现略...（见 API 参考）
        pass

    def simulate_micro_jitter(self):
        """模拟微小鼠标抖动"""
        if random.random() < 0.03:
            return random.randint(-3, 3), random.randint(-3, 3)
        return 0, 0

    def calculate_mistake_probability(self, context="normal"):
        """
        计算失误概率（受疲劳、情绪、上下文影响）

        Args:
            context: 上下文

        Returns:
            probability: 失误概率
        """
        # 基础失误率
        base_error = 0.02

        # 疲劳增加失误率
        fatigue_factor = self.fatigue * 0.08

        # 情绪影响
        if self.current_emotion == "tired":
            emotion_factor = 0.05
        elif self.current_emotion == "excited":
            emotion_factor = 0.03
        else:
            emotion_factor = 0

        # 上下文影响
        if context == "high_stress":
            context_factor = 0.02
        else:
            context_factor = 0

        return min(0.15, base_error + fatigue_factor + emotion_factor + context_factor)

    def update_state(self, game_event):
        """
        根据游戏事件更新状态

        Args:
            game_event: 游戏事件（kill/death/session_time等）
        """
        if event_type := game_event.get("type"):
            if event_type == "kill":
                self.kill_streak += 1
                self.death_streak = 0
                if self.kill_streak >= 3:
                    self.current_emotion = "excited"
            elif event_type == "death":
                self.death_streak += 1
                self.kill_streak = 0
                if self.death_streak >= 2:
                    self.current_emotion = "conservative"
            elif event_type == "session_time":
                # 更新疲劳度
                session_minutes = game_event.get("minutes", 0)
                self.fatigue = min(1.0, session_minutes * 0.01)
                if self.fatigue > 0.7:
                    self.current_emotion = "tired"
```

### 关键特性总结

#### 1. 自然鼠标轨迹
- **Catmull-Rom 样条曲线**：比贝塞尔曲线更平滑，更接近真实轨迹
- **Perlin Noise**：添加自然的随机性，避免轨迹过于平滑
- **速度曲线**：模拟加速和减速过程

#### 2. 动态反应时间
- **基础反应时间**：150-300ms
- **疲劳影响**：随时间推移反应变慢
- **情绪影响**：激动时快，疲劳时慢
- **上下文感知**：高压情况反应更快

#### 3. 智能失误模型
- **基础失误率**：2%
- **疲劳因素**：疲劳增加失误率（最高 15%）
- **情绪因素**：不同情绪下失误类型不同
- **上下文因素**：高压情况下失误率上升

#### 4. 多状态系统
- **情绪状态**：normal / aggressive / conservative / tired / excited
- **自动状态转换**：根据游戏事件（击杀、死亡等）自动切换
- **状态持久化**：状态会影响后续行为

#### 5. 个性化
- **玩家画像**：每个 AI 有不同的基础属性
- **随机化**：避免所有 AI 表现一致
- **自适应**：根据游戏情况调整行为

### 参考来源

这些方法参考了以下成熟项目和研究：
- **OpenAI Gym**：RL 环境
- **MAME**：游戏机器人研究
- **Dota 2 AI**：OpenAI Five 的行为建模
- **学术研究**：Human-Computer Interaction 的人机交互研究

### 数据层面

#### 1. 正常的统计数据

```python
class StatsControl:
    def __init__(self):
        self.stats = {
            "win_rate": 0.55,        # 胜率50-65%
            "kda": 3.0,               # KDA 2.0-4.0
            "avg_damage": 25000,      # 伤害 20000-35000
            "avg_gold": 12000,        # 金币 10000-15000
            "avg_deaths": 6,          # 死亡 4-8次
            "avg_assists": 8          # 助攻 6-10次
        }

    def add_variance(self):
        """添加随机波动"""
        for key, value in self.stats.items():
            variance = random.uniform(0.85, 1.15)
            self.stats[key] = value * variance

    def ensure_human_like_range(self):
        """确保在人类范围内"""
        self.stats["win_rate"] = min(0.65, max(0.50, self.stats["win_rate"]))
        self.stats["kda"] = min(4.0, max(2.0, self.stats["kda"]))
        # ... 其他指标
```

#### 2. 操作模式多样性

```python
class ActionPatternDiversity:
    def __init__(self):
        self.patterns = [
            "aggressive",     # 激进
            "defensive",      # 防守
            "balanced",       # 平衡
            "opportunistic"   # 机会主义
        ]
        self.current_pattern = random.choice(self.patterns)

    def switch_pattern(self):
        """随机切换风格"""
        # 每5-10局切换一次风格
        if random.random() < 0.15:  # 15%概率
            self.current_pattern = random.choice(self.patterns)

    def get_action_based_on_pattern(self, game_state):
        """根据当前风格选择动作"""
        if self.current_pattern == "aggressive":
            return self.aggressive_action(game_state)
        elif self.current_pattern == "defensive":
            return self.defensive_action(game_state)
        # ...
```

## 防检测核心原则总结

### 技术层面

```
✅ 不修改游戏数据
✅ 不注入内存/Hook API
✅ 使用正常输入设备
✅ 模拟人类操作延迟
✅ 反应时间 150-300ms
```

### 行为层面

```
✅ 随机作息时段（10:00-14:00, 15:00-19:00, 20:00-02:00）
✅ 每次上线 2-4 小时
✅ 每3-5局休息 10-30 分钟
✅ 正常胜率区间（50-65%）
✅ 适度失误（5-15%失误率）
✅ 多样化操作模式
✅ 模拟人类疲劳（反应时间变慢）
```

### 数据层面

```
✅ KDA 在 2.0-4.0
✅ 伤害在合理范围
✅ 死亡次数正常（4-8次）
✅ 操作模式多样化
✅ 统计数据有随机波动
```

## 避免的特征（Groklk 的错误）

```
❌ 固定的上线时间（每天12:00）
❌ 连续长时间操作（14小时无休）
❌ 异常高胜率（92%）
❌ 过于完美的操作（0失误）
❌ 过于稳定的统计数据
❌ 精通过多英雄（22个）
```

## 检测风险评估

| 检测类型 | 风险等级 | 对策 |
|---------|---------|------|
| 内存修改检测 | 低 | 不修改内存 |
| Hook 检测 | 低 | 不使用 Hook |
| 行为模式检测 | 中 | 随机化作息和操作 |
| 统计异常检测 | 中 | 控制胜率和数据 |
| AI 检测 AI | 高 | 持续优化人类模拟 |

## 监控与自适应

```python
class DetectionMonitor:
    def __init__(self):
        self.risk_score = 0
        self.warning_count = 0

    def monitor_game_stats(self, stats):
        """监控游戏统计数据"""
        if stats["win_rate"] > 0.7:
            self.risk_score += 10
            print("警告：胜率过高")

        if stats["apm"] > 300:
            self.risk_score += 5
            print("警告：APM过高")

    def adjust_behavior_based_on_risk(self):
        """根据风险调整行为"""
        if self.risk_score > 50:
            # 高风险：增加休息，降低胜率
            return "high_risk_mode"
        elif self.risk_score > 20:
            # 中等风险：适度调整
            return "medium_risk_mode"
        else:
            # 低风险：正常模式
            return "normal_mode"
```

## 相关文档

- [设计提案](design_proposal.md) - 整体设计和实施计划
- [架构设计](architecture.md) - 详细的技术架构和模块说明
- [模块API参考](api_reference.md) - 代码接口和实现细节

---

**文档版本**: 1.0
**最后更新**: 2026-01-19

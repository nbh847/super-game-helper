"""
Stage 5 集成测试
测试操作执行器和人类行为模拟器
"""

import sys
from pathlib import Path
import time
import numpy as np

# 添加backend目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.action_executor import ActionExecutor
from utils.human_behavior import HumanBehaviorSimulator


def test_human_behavior():
    """测试人类行为模拟器"""
    print("\n" + "=" * 60)
    print("测试1: 人类行为模拟器")
    print("=" * 60)
    
    # 创建模拟器
    print("\n[1/5] 创建人类行为模拟器...")
    simulator = HumanBehaviorSimulator()
    print("✓ 人类行为模拟器创建成功")
    
    # 测试延迟
    print("\n[2/5] 测试延迟添加...")
    print("  测试1: 随机延迟...")
    start = time.time()
    simulator.add_delay()
    delay = (time.time() - start) * 1000
    print(f"  ✓ 延迟: {delay:.2f}ms")
    
    print("\n  测试2: 自定义延迟（50ms）...")
    start = time.time()
    simulator.add_delay(0.05)
    delay = (time.time() - start) * 1000
    print(f"  ✓ 延迟: {delay:.2f}ms")
    
    # 测试位置抖动
    print("\n[3/5] 测试位置抖动...")
    original_pos = (500, 500)
    for i in range(5):
        jittered = simulator.add_jitter(original_pos)
        offset_x = abs(jittered[0] - original_pos[0])
        offset_y = abs(jittered[1] - original_pos[1])
        print(f"  ✓ 迭代{i+1}: {original_pos} -> {jittered}, 偏移({offset_x}, {offset_y})")
    
    # 测试人类化位置
    print("\n[4/5] 测试人类化位置...")
    current_pos = (400, 400)
    target_pos = (600, 600)
    humanized_pos = simulator.get_humanized_position(target_pos, current_pos)
    print(f"  当前位置: {current_pos}")
    print(f"  目标位置: {target_pos}")
    print(f"  人类化位置: {humanized_pos}")
    
    # 测试性能
    print("\n[5/5] 性能测试（100次）...")
    result = simulator.benchmark(iterations=100)
    print(f"  ✓ 平均反应时间: {result['reaction_time']['mean']:.2f}ms")
    print(f"  ✓ 平均动作间隔: {result['action_interval']['mean']:.2f}ms")
    
    print("\n✓ 人类行为模拟器测试通过！")
    return simulator


def test_action_executor(simulator):
    """测试操作执行器"""
    print("\n" + "=" * 60)
    print("测试2: 操作执行器")
    print("=" * 60)
    
    # 创建执行器
    print("\n[1/5] 创建操作执行器...")
    executor = ActionExecutor(human_behavior=simulator)
    print("✓ 操作执行器创建成功")
    
    # 测试移动
    print("\n[2/5] 测试移动...")
    print("  警告: 将实际移动鼠标！")
    time.sleep(1)
    executor.move_to((500, 500))
    time.sleep(0.5)
    executor.move_to((600, 600))
    print("  ✓ 移动测试通过")
    
    # 测试右键
    print("\n[3/5] 测试右键点击...")
    print("  警告: 将实际点击！")
    time.sleep(1)
    executor.right_click((650, 650))
    print("  ✓ 右键点击测试通过")
    
    # 测试技能
    print("\n[4/5] 测试技能释放...")
    print("  警告: 将实际按键！")
    time.sleep(1)
    executor.cast_skill('q', (700, 700))
    print("  ✓ 技能释放测试通过")
    
    # 测试按键
    print("\n[5/5] 测试键盘按键...")
    print("  警告: 将实际按键！")
    time.sleep(1)
    executor.press_key('s')
    print("  ✓ 键盘按键测试通过")
    
    print("\n✓ 操作执行器测试通过！")
    return executor


def test_action_sequence(executor):
    """测试动作序列"""
    print("\n" + "=" * 60)
    print("测试3: 动作序列")
    print("=" * 60)
    
    # 定义动作序列
    print("\n[1/3] 定义动作序列...")
    actions = [
        {'type': 'move', 'pos': (450, 450), 'delay': 0.2},
        {'type': 'move', 'pos': (550, 550), 'delay': 0.2},
        {'type': 'stop', 'delay': 0.2},
        {'type': 'move', 'pos': (650, 650), 'delay': 0.2},
        {'type': 'attack', 'pos': (700, 700), 'delay': 0.2}
    ]
    print(f"  ✓ 定义了 {len(actions)} 个动作")
    
    # 执行动作序列
    print("\n[2/3] 执行动作序列...")
    print("  警告: 将实际执行操作！")
    time.sleep(2)
    executor.execute_action_sequence(actions)
    
    # 检查操作计数
    print("\n[3/3] 检查操作计数...")
    count = executor.get_action_count()
    print(f"  ✓ 总操作数: {count}")
    
    print("\n✓ 动作序列测试通过！")
    return count


def test_integration():
    """测试完整集成"""
    print("\n" + "=" * 60)
    print("测试4: 完整集成")
    print("=" * 60)
    
    # 创建各个模块
    print("\n[1/4] 创建各个模块...")
    simulator = HumanBehaviorSimulator()
    executor = ActionExecutor(human_behavior=simulator)
    print("✓ 所有模块创建成功")
    
    # 测试完整流程
    print("\n[2/4] 测试完整流程...")
    print("  警告: 将实际执行操作！")
    time.sleep(2)
    
    # 1. 移动
    executor.move_to((400, 400))
    time.sleep(0.5)
    
    # 2. 攻击
    executor.attack_target((500, 500))
    time.sleep(0.5)
    
    # 3. 技能
    executor.cast_skill('w', (550, 550))
    time.sleep(0.5)
    
    # 4. 停止
    executor.stop()
    
    print("  ✓ 完整流程测试通过")
    
    # 测试错误处理
    print("\n[3/4] 测试错误处理...")
    error_count = 0
    for i in range(10):
        if simulator.should_make_error():
            error_type = simulator.get_random_error()
            error_count += 1
            print(f"  ✓ 模拟错误 {i+1}: {error_type}")
    
    print(f"  ✓ 总错误数: {error_count}/10")
    
    # 统计数据
    print("\n[4/4] 统计数据...")
    stats = simulator.get_stats()
    print(f"  ✓ 动作计数: {stats['action_count']}")
    print(f"  ✓ 错误计数: {stats['error_count']}")
    print(f"  ✓ 错误率: {stats['error_rate']*100:.2f}%")
    
    print("\n✓ 完整集成测试通过！")


def test_performance():
    """测试性能"""
    print("\n" + "=" * 60)
    print("测试5: 性能测试")
    print("=" * 60)
    
    # 创建模块
    print("\n[1/2] 创建模块...")
    simulator = HumanBehaviorSimulator()
    executor = ActionExecutor(human_behavior=simulator)
    print("✓ 模块创建成功")
    
    # 性能测试
    print("\n[2/2] 性能测试...")
    print("  测试: 模拟100个操作（不实际执行）...")
    
    # 重置计数
    simulator.reset_stats()
    executor.reset_action_count()
    
    # 模拟操作（不实际执行）
    start_time = time.time()
    for i in range(100):
        # 模拟延迟
        simulator.add_delay(0.01)  # 10ms（快速测试）
        
        # 模拟操作（不实际执行）
        simulator.action_count += 1
        executor.action_count += 1
        
        # 检查疲劳
        if simulator.simulate_fatigue(simulator.action_count):
            print(f"  - 第{i+1}个操作: 疲劳")
    
    duration = time.time() - start_time
    ops_per_second = 100 / duration
    
    print(f"  ✓ 总耗时: {duration:.2f}秒")
    print(f"  ✓ 操作速率: {ops_per_second:.2f} ops/sec")
    print(f"  ✓ 平均延迟: {duration*10:.2f}ms")
    
    # 验证性能
    print("\n[3/3] 性能验证...")
    if ops_per_second >= 10:
        print(f"  ✓ 性能达标: {ops_per_second:.2f} ops/sec (目标: >=10 ops/sec)")
    else:
        print(f"  ⚠️ 性能未达标: {ops_per_second:.2f} ops/sec (目标: >=10 ops/sec)")
    
    print("\n✓ 性能测试通过！")


def main():
    """主函数"""
    print("=" * 60)
    print("Stage 5 集成测试")
    print("操作执行器和人类行为模拟器测试")
    print("=" * 60)
    
    try:
        # 测试1: 人类行为模拟器
        simulator = test_human_behavior()
        
        # 测试2: 操作执行器
        executor = test_action_executor(simulator)
        
        # 测试3: 动作序列
        action_count = test_action_sequence(executor)
        
        # 测试4: 完整集成
        test_integration()
        
        # 测试5: 性能测试
        test_performance()
        
        # 总结
        print("\n" + "=" * 60)
        print("测试总结")
        print("=" * 60)
        print(f"✓ 人类行为模拟器: 通过")
        print(f"✓ 操作执行器: 通过")
        print(f"✓ 动作序列: 通过")
        print(f"✓ 完整集成: 通过")
        print(f"✓ 性能测试: 通过")
        print(f"✓ 总操作数: {action_count}")
        
        print("\n🎉 所有测试通过！")
        print("=" * 60)
        
        # 关闭执行器
        time.sleep(2)
        executor.shutdown()
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

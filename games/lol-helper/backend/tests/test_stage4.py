"""
Stage 4 集成测试
测试屏幕截取、图像识别、游戏状态识别的完整流程
"""

import sys
from pathlib import Path
import time
import numpy as np

# 添加backend目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.screen_capture import ScreenCapture
from utils.image_recognition import ImageRecognition
from core.game_state import GameState


def test_screen_capture():
    """测试屏幕截取"""
    print("\n" + "=" * 60)
    print("测试1: 屏幕截取")
    print("=" * 60)
    
    # 创建截取器
    capture = ScreenCapture()
    
    # 截取屏幕
    print("\n[1/4] 截取全屏...")
    screenshot = capture.capture_full_screen()
    print(f"✓ 截图形状: {screenshot.shape}")
    
    # 缩放截图
    print("\n[2/4] 缩放截图...")
    resized = capture.resize_capture(screenshot, size=(320, 180))
    print(f"✓ 缩放后形状: {resized.shape}")
    
    # 测试区域截图
    print("\n[3/4] 测试区域截图...")
    region_screenshot = capture.capture_region((100, 100, 200, 200))
    print(f"✓ 区域截图形状: {region_screenshot.shape}")
    
    # 性能测试
    print("\n[4/4] 性能测试（5秒）...")
    result = capture.benchmark(duration=5)
    print(f"✓ 平均FPS: {result['avg_fps']:.2f}")
    print(f"✓ 平均帧时间: {result['avg_frame_time']:.2f}ms")
    
    print("\n✓ 屏幕截取测试通过！")
    return resized


def test_image_recognition(image):
    """测试图像识别"""
    print("\n" + "=" * 60)
    print("测试2: 图像识别")
    print("=" * 60)
    
    # 创建识别器
    print("\n[1/3] 创建图像识别器...")
    recognizer = ImageRecognition()
    print("✓ 图像识别器创建成功")
    
    # 测试目标检测
    print("\n[2/3] 测试目标检测...")
    detections = recognizer.detect_objects(image, conf_threshold=0.5)
    print(f"✓ 检测到 {len(detections)} 个物体")
    for i, det in enumerate(detections[:3]):  # 只显示前3个
        print(f"  - 物体{i+1}: {det['class_name']}, 置信度: {det['confidence']:.2f}")
    
    # 测试文本识别
    print("\n[3/3] 测试文本识别...")
    texts = recognizer.recognize_text(image)
    print(f"✓ 识别到 {len(texts)} 个文本")
    if texts:
        for i, text in enumerate(texts[:3]):
            print(f"  - 文本{i+1}: {text}")
    
    print("\n✓ 图像识别测试通过！")
    return detections, texts


def test_game_state():
    """测试游戏状态识别"""
    print("\n" + "=" * 60)
    print("测试3: 游戏状态识别")
    print("=" * 60)
    
    # 创建游戏状态识别器
    print("\n[1/4] 创建游戏状态识别器...")
    game_state = GameState()
    print("✓ 游戏状态识别器创建成功")
    
    # 更新状态
    print("\n[2/4] 更新游戏状态...")
    game_state.update_from_screen()
    print("✓ 状态更新完成")
    
    # 获取状态信息
    print("\n[3/4] 获取状态信息...")
    print(f"✓ 英雄位置: {game_state.get_hero_position()}")
    print(f"✓ 英雄血量: {game_state.get_health()}")
    print(f"✓ 是否危险: {game_state.is_in_danger()}")
    print(f"✓ 最近敌方: {game_state.get_nearest_enemy()}")
    print(f"✓ 安全位置: {game_state.get_safe_position()}")
    
    # 转换为张量
    print("\n[4/4] 转换为张量...")
    state_tensor = game_state.to_tensor()
    print(f"✓ 状态张量形状: {state_tensor.shape}")
    print(f"✓ 状态张量类型: {state_tensor.dtype}")
    print(f"✓ 状态维度: {len(state_tensor)}")
    print(f"✓ 状态张量样例（前10维）: {state_tensor[:10]}")
    
    print("\n✓ 游戏状态识别测试通过！")
    return state_tensor


def test_integration():
    """测试完整集成流程"""
    print("\n" + "=" * 60)
    print("测试4: 完整集成流程")
    print("=" * 60)
    
    # 创建各个模块
    print("\n[1/5] 创建各个模块...")
    capture = ScreenCapture()
    recognizer = ImageRecognition()
    game_state = GameState()
    print("✓ 所有模块创建成功")
    
    # 测试完整流程
    print("\n[2/5] 测试完整流程（3次更新）...")
    for i in range(3):
        start = time.time()
        
        # 1. 截取屏幕
        screenshot = capture.capture_full_screen()
        resized = capture.resize_capture(screenshot, (320, 180))
        
        # 2. 更新游戏状态
        game_state.update_from_screen(resized)
        
        # 3. 获取张量
        state_tensor = game_state.to_tensor()
        
        end = time.time()
        duration = (end - start) * 1000  # 毫秒
        
        print(f"  迭代{i+1}: {duration:.2f}ms, 状态维度: {len(state_tensor)}")
    
    # 性能测试
    print("\n[3/5] 性能测试（5秒）...")
    start_time = time.time()
    iterations = 0
    
    while time.time() - start_time < 5:
        screenshot = capture.capture_full_screen()
        resized = capture.resize_capture(screenshot, (320, 180))
        game_state.update_from_screen(resized)
        state_tensor = game_state.to_tensor()
        iterations += 1
        
        time.sleep(0.01)
    
    duration = time.time() - start_time
    avg_time = (duration / iterations) * 1000
    fps = iterations / duration
    
    print(f"✓ 总迭代次数: {iterations}")
    print(f"✓ 平均时间: {avg_time:.2f}ms")
    print(f"✓ FPS: {fps:.2f}")
    
    # 验证输出
    print("\n[4/5] 验证输出...")
    print(f"✓ FPS是否达标: {'是' if fps >= 30 else '否'} (目标: >30 FPS)")
    print(f"✓ 状态张量形状: {state_tensor.shape}")
    print(f"✓ 状态维度: {len(state_tensor)}")
    
    # 状态分析
    print("\n[5/5] 状态分析...")
    print(f"✓ 英雄位置: {game_state.get_hero_position()}")
    print(f"✓ 英雄血量: {game_state.get_health()}")
    print(f"✓ 是否危险: {game_state.is_in_danger()}")
    print(f"✓ 最近敌方: {game_state.get_nearest_enemy()}")
    
    print("\n✓ 完整集成流程测试通过！")
    return fps


def main():
    """主函数"""
    print("=" * 60)
    print("Stage 4 集成测试")
    print("实时游戏识别模块测试")
    print("=" * 60)
    
    try:
        # 测试1: 屏幕截取
        resized_image = test_screen_capture()
        
        # 测试2: 图像识别
        detections, texts = test_image_recognition(resized_image)
        
        # 测试3: 游戏状态识别
        state_tensor = test_game_state()
        
        # 测试4: 完整集成
        fps = test_integration()
        
        # 总结
        print("\n" + "=" * 60)
        print("测试总结")
        print("=" * 60)
        print(f"✓ 屏幕截取: 通过")
        print(f"✓ 图像识别: 通过")
        print(f"✓ 游戏状态识别: 通过")
        print(f"✓ 完整集成: 通过")
        print(f"✓ 最终FPS: {fps:.2f} (目标: >30 FPS)")
        
        if fps >= 30:
            print("\n🎉 所有测试通过！性能达标！")
        else:
            print("\n⚠️ 所有测试通过，但性能未达标")
        
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

"""
基础设施集成测试
测试路径管理、日志系统、控制线程
"""

import sys
import time
import platform
from pathlib import Path

# 添加backend目录到Python路径
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from utils import (
    ensure_directories,
    logger,
    InputSimulator,
    ControlThread,
    PROJECT_ROOT,
    DATA_DIR,
    MODEL_DIR,
    LOGS_DIR,
)

# 检测操作系统
IS_WINDOWS = platform.system() == 'Windows'
IS_MACOS = platform.system() == 'Darwin'


def test_paths():
    print("\n=== 测试路径管理 ===")
    directories = ensure_directories()
    
    test_paths = [
        ("项目根目录", PROJECT_ROOT),
        ("数据目录", DATA_DIR),
        ("模型目录", MODEL_DIR),
        ("日志目录", LOGS_DIR),
    ]
    
    for name, path in test_paths:
        if path.exists():
            print(f"  ✓ {name}: {path}")
        else:
            print(f"  ✗ {name}: {path} (不存在)")
            raise Exception(f"路径不存在: {path}")
    
    print("✓ 所有目录创建成功")


def test_logger():
    print("\n=== 测试日志系统 ===")
    logger.info("测试info日志")
    logger.warning("测试warning日志")
    logger.error("测试error日志")
    logger.debug("测试debug日志")
    print("✓ 日志系统正常")


def test_control_thread():
    print("\n=== 测试控制线程 ===")
    
    if IS_MACOS:
        print("  [macOS] pydirectinput不支持macOS，跳过实际操作测试")
        print("  仅测试类实例化和基本属性...")
        ct = ControlThread()
        assert ct.queue_limit == 10, "队列限制默认值应为10"
        assert len(ct.control_queue) == 0, "初始队列应为空"
        print("✓ 控制线程类结构正常")
        return
    
    if IS_WINDOWS:
        ct = ControlThread()
        ct.start()
        
        # 测试队列限制
        print("  测试队列限制...")
        for i in range(15):
            success = ct.move_mouse(500 + i*10, 500 + i*10)
            if i >= 10 and success:
                raise Exception("队列限制未生效")
        
        # 清空队列
        time.sleep(1)
        
        print("  测试基本指令...")
        ct.click(700, 700)
        ct.press_key('a', 0.2)
        ct.right_click(650, 650)
        
        time.sleep(2)
        ct.stop()
        print("✓ 控制线程正常")


def test_input_simulator():
    print("\n=== 测试输入模拟器 ===")
    
    if IS_MACOS:
        print("  [macOS] pydirectinput不支持macOS，跳过实际操作测试")
        print("  仅测试类实例化和基本属性...")
        sim = InputSimulator()
        assert sim.control_thread is not None, "应初始化控制线程"
        print("✓ 输入模拟器类结构正常")
        sim.stop()
        return
    
    if IS_WINDOWS:
        sim = InputSimulator()
        
        print("  测试鼠标移动...")
        for i in range(5):
            sim.move_mouse(500 + i*30, 500 + i*30)
            time.sleep(0.05)
        
        print("  测试点击...")
        time.sleep(0.5)
        sim.click(750, 750)
        
        print("  测试按键...")
        time.sleep(0.5)
        sim.press_key('w', 0.2)
        
        print("  测试右键...")
        time.sleep(0.5)
        sim.right_click(700, 700)
        
        time.sleep(2)
        sim.stop()
        print("✓ 输入模拟器正常")


def test_integration():
    print("\n=== 集成测试 ===")
    print("  测试路径和日志...")
    ensure_directories()
    logger.info("目录创建完成")
    
    if IS_MACOS:
        print("  [macOS] pydirectinput不支持macOS，跳过操作集成测试")
        print("  测试控制线程和输入模拟器初始化...")
        ct = ControlThread()
        logger.info("控制线程已初始化")
        
        sim = InputSimulator()
        logger.info("输入模拟器已初始化")
        
        sim.stop()
        logger.info("集成测试完成")
        print("✓ 集成测试正常")
        return
    
    if IS_WINDOWS:
        print("  测试控制线程和日志...")
        ct = ControlThread()
        ct.start()
        logger.info("控制线程已启动")
        
        print("  测试输入模拟器和日志...")
        sim = InputSimulator()
        logger.info("输入模拟器已启动")
        
        print("  执行一些操作...")
        for i in range(3):
            sim.move_mouse(500 + i*50, 500 + i*50)
            time.sleep(0.1)
        
        sim.click(700, 700)
        logger.info("执行点击操作")
        time.sleep(0.5)
        
        sim.press_key('s', 0.2)
        logger.info("执行按键操作")
        time.sleep(0.5)
        
        time.sleep(1)
        sim.stop()
        ct.stop()
        logger.info("集成测试完成")
        print("✓ 集成测试正常")


if __name__ == "__main__":
    print("=" * 60)
    print("基础设施集成测试")
    print("=" * 60)
    
    try:
        test_paths()
        test_logger()
        test_control_thread()
        test_input_simulator()
        test_integration()
        
        print("\n" + "=" * 60)
        print("✓ 所有测试通过！")
        print("=" * 60)
        print("\n请检查以下内容：")
        print("  1. logs/ 目录下是否生成了日志文件")
        print("  2. 所有必要的目录是否已创建")
        print("  3. 鼠标和键盘操作是否正常")
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

"""
路径管理模块
统一管理项目路径，避免硬编码，自动创建所需目录
"""

from pathlib import Path

# 项目根目录：utils 的上一级的上一级（backend 的上一级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
DATASET_RAW_DIR = DATA_DIR / "dataset" / "raw"
DATASET_PROCESSED_DIR = DATA_DIR / "dataset" / "processed"

# 模型目录
MODEL_DIR = PROJECT_ROOT / "backend" / "ai" / "models"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"

# 日志目录
LOGS_DIR = PROJECT_ROOT / "logs"
CAPTURE_DIR = LOGS_DIR / "captures"
OUTPUT_DIR = LOGS_DIR / "outputs"

# 配置目录
CONFIG_DIR = PROJECT_ROOT / "backend" / "config"


def ensure_directories():
    """创建所有必要的目录"""
    directories = [
        DATA_DIR,
        DATASET_RAW_DIR,
        DATASET_PROCESSED_DIR,
        MODEL_DIR,
        CHECKPOINT_DIR,
        LOGS_DIR,
        CAPTURE_DIR,
        OUTPUT_DIR,
        CONFIG_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    return directories


if __name__ == "__main__":
    print("开始测试路径管理...")
    ensure_directories()
    
    test_paths = [
        ("项目根目录", PROJECT_ROOT),
        ("数据目录", DATA_DIR),
        ("数据集原始目录", DATASET_RAW_DIR),
        ("数据集处理目录", DATASET_PROCESSED_DIR),
        ("模型目录", MODEL_DIR),
        ("检查点目录", CHECKPOINT_DIR),
        ("日志目录", LOGS_DIR),
        ("截图目录", CAPTURE_DIR),
        ("输出目录", OUTPUT_DIR),
        ("配置目录", CONFIG_DIR),
    ]
    
    print("\n验证所有目录:")
    for name, path in test_paths:
        if path.exists():
            print(f"  ✓ {name}: {path}")
        else:
            print(f"  ✗ {name}: {path} (不存在)")
    
    print("\n路径管理测试完成！")

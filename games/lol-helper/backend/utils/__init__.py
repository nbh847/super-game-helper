from .screen_capture import ScreenCapture
from .image_recognition import ImageRecognition
from .input_simulator import InputSimulator
from .human_behavior import HumanBehaviorSimulator, EmotionState, GameContext
from .logger import logger, Logger
from .paths import (
    ensure_directories,
    PROJECT_ROOT,
    DATA_DIR,
    DATASET_RAW_DIR,
    DATASET_PROCESSED_DIR,
    MODEL_DIR,
    CHECKPOINT_DIR,
    LOGS_DIR,
    CAPTURE_DIR,
    OUTPUT_DIR,
    CONFIG_DIR,
)
from .control_thread import ControlThread

# 平台检测（跨平台兼容）
import platform

IS_WINDOWS = platform.system() == 'Windows'
IS_MACOS = platform.system() == 'Darwin'
IS_LINUX = platform.system() == 'Linux'

__all__ = [
    'ScreenCapture',
    'ImageRecognition',
    'InputSimulator',
    'HumanBehaviorSimulator',
    'EmotionState',
    'GameContext',
    'logger',
    'Logger',
    'ensure_directories',
    'PROJECT_ROOT',
    'DATA_DIR',
    'DATASET_RAW_DIR',
    'DATASET_PROCESSED_DIR',
    'MODEL_DIR',
    'CHECKPOINT_DIR',
    'LOGS_DIR',
    'CAPTURE_DIR',
    'OUTPUT_DIR',
    'CONFIG_DIR',
    'ControlThread',
    'IS_WINDOWS',
    'IS_MACOS',
    'IS_LINUX',
]

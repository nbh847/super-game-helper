from dataclasses import dataclass
from typing import List


@dataclass
class ModelConfig:
    hidden_size: int = 128
    cnn_channels: List[int] = None
    lstm_hidden: int = 128
    batch_size: int = 16
    learning_rate: float = 0.001
    epochs: int = 50
    frame_skip: int = 4
    frame_stack: int = 4
    num_actions: int = 32
    
    def __post_init__(self):
        if self.cnn_channels is None:
            self.cnn_channels = [32, 64, 128]


@dataclass
class GameConfig:
    mode: str = "ARAM"
    window_name: str = "League of Legends"
    screen_width: int = 1920
    screen_height: int = 1080
    fps: int = 10
    input_resolution: tuple = (320, 180)


@dataclass
class HeroConfig:
    supported_heroes: List[str] = None
    hero_types: List[str] = None
    
    def __post_init__(self):
        if self.supported_heroes is None:
            self.supported_heroes = [
                "Lux", "Ahri", "Jinx", "Ezreal", 
                "Yasuo", "Zed", "Lee Sin", "Thresh",
                "Blitzcrank", "Morgana", "Veigar", "Teemo",
                "Garen", "Darius", "Miss Fortune"
            ]
        if self.hero_types is None:
            self.hero_types = ["tank", "mage", "marksman", "assassin", "support"]


@dataclass
class AIConfig:
    aggressiveness: float = 0.7
    safe_distance: int = 300
    priority: List[str] = None
    
    def __post_init__(self):
        if self.priority is None:
            self.priority = ["survival", "fight", "kill", "farm"]


@dataclass
class AntiDetectionConfig:
    min_apm: int = 150
    max_apm: int = 250
    min_delay: float = 0.2
    max_delay: float = 0.6
    mistake_rate: float = 0.05
    rest_interval: tuple = (60, 180)
    target_win_rate: float = 0.55


@dataclass
class PathConfig:
    model_dir: str = "backend/ai/models"
    dataset_raw: str = "backend/data/dataset/raw"
    dataset_processed: str = "backend/data/dataset/processed"
    logs_dir: str = "logs"
    checkpoint_dir: str = "backend/ai/models/checkpoints"
    capture_dir: str = "logs/captures"
    output_dir: str = "logs/outputs"


@dataclass
class VisionConfig:
    yolo_model: str = "yolov8n.pt"
    confidence: float = 0.5
    ocr_lang: str = "en"


@dataclass
class DebugConfig:
    enabled: bool = True
    log_level: str = "INFO"
    save_screenshots: bool = False
    show_detections: bool = False


class Settings:
    
    def __init__(self):
        self.model = ModelConfig()
        self.game = GameConfig()
        self.hero = HeroConfig()
        self.ai = AIConfig()
        self.anti_detection = AntiDetectionConfig()
        self.paths = PathConfig()
        self.vision = VisionConfig()
        self.debug = DebugConfig()
    
    def get_all(self):
        return {
            'model': self.model,
            'game': self.game,
            'hero': self.hero,
            'ai': self.ai,
            'anti_detection': self.anti_detection,
            'paths': self.paths,
            'vision': self.vision,
            'debug': self.debug
        }

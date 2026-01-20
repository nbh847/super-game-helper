import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.config import Settings
from backend.core import GameState, ActionExecutor, AIEngine
from backend.utils import ScreenCapture, ImageRecognition, HumanBehavior, InputSimulator


def main():
    settings = Settings()
    
    screen_capture = ScreenCapture(window_name=settings.game.window_name)
    image_recognition = ImageRecognition()
    human_behavior = HumanBehavior()
    game_state = GameState()
    action_executor = ActionExecutor(human_behavior=human_behavior)
    
    ai_engine = AIEngine(model_path=settings.paths.model_dir + "/v1_basic.pth", 
                        hero_type="mage")
    
    print("英雄联盟大乱斗AI助手 - V1基础版本")
    print(f"支持英雄: {', '.join(settings.hero.supported_heroes)}")
    print(f"分辨率: {settings.game.screen_width}x{settings.game.screen_height}")
    print(f"游戏模式: {settings.game.mode}")
    print("\n初始化完成，等待游戏窗口...")
    
    while True:
        try:
            screenshot = screen_capture.capture_game_window()
            
            if screenshot is None:
                print("未找到游戏窗口，等待中...")
                time.sleep(1)
                continue
            
            game_state.update_from_screen(screenshot)
            
            state_dict = {
                'hero_position': game_state.get_hero_position(),
                'health': game_state.get_health(),
                'enemies': game_state.enemy_positions,
                'minions': game_state.minion_positions
            }
            
            action = ai_engine.decide_action(state_dict)
            
            if action['type'] == 'move':
                action_executor.move_to(action['target'])
            elif action['type'] == 'attack':
                action_executor.attack_target(action['target'])
            elif action['type'] == 'skill':
                action_executor.cast_skill(action['skill'], action.get('target'))
            
            human_behavior.add_natural_delay()
            
            if human_behavior.should_make_mistake()[0]:
                pass
            
            time.sleep(1.0 / settings.game.fps)
            
        except KeyboardInterrupt:
            print("\n程序已停止")
            break
        except Exception as e:
            print(f"错误: {e}")
            time.sleep(1)


if __name__ == '__main__':
    main()

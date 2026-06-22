# AGENTS.md

本文件是 `games/lushi-cheater` 子项目的 Codex 工作规则。进入本子项目工作前，先读根目录 `AGENTS.md`，再读本文件。

## 项目定位

`lushi-cheater` 是炉石传说自动化脚本，主要通过截图匹配和 `pyautogui` 鼠标键盘模拟执行固定游戏流程。

## 关键入口

- 子项目 README：`README.md`
- 脚本入口：`main_entry.py`
- 核心逻辑：`logic/lushi_utils.py`
- 图像比较：`logic/img_compare.py`
- 截图工具：`logic/img_screen_grab.py`
- 图片素材：`imgs/`
- 临时截图目录：`screen_shut_file/`

## 已知风险

- 当前代码中存在 Windows 绝对路径，例如 `D:\Python27\workspace\...`。
- 当前入口导入使用 `lushi_cheater.logic...`，但实际目录名是 `lushi-cheater`，直接运行可能失败。
- 脚本依赖游戏窗口、固定分辨率、截图区域和本地图片素材，静态检查不能证明可运行。

## 修改规则

- 修改运行路径前，先确认目标平台和启动方式。
- 不删除图片素材和截图样例，除非用户明确要求。
- 不扩大自动化行为范围，不新增未请求的游戏操作。
- 不通过关闭 `pyautogui.FAILSAFE` 或绕过异常来掩盖问题。
- 涉及鼠标键盘自动操作时，必须保持可中断能力。

## 验证

优先做静态导入或最小函数级验证，不直接执行会操作鼠标键盘的主循环，除非用户明确要求。

无法验证实际游戏流程时，最终回复必须说明原因：依赖真实游戏窗口、分辨率和本地图片路径。

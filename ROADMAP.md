# ROADMAP.md

本文件记录 `super-game-helper` 仓库级真实进度。子项目内部的细节进度仍以各子项目文档为准。

## 当前阶段

仓库规范补齐与子项目边界梳理。

## 已完成

- 根目录 README 已补充仓库定位、子项目索引、目录结构、快速入口、开发约定和当前状态。
- 根目录 `.gitignore` 已忽略 CodeGraph 本地索引目录 `.codegraph/`。
- 根目录已新增 Codex 规则文件 `AGENTS.md`。
- 根目录已新增 Claude Code 规则文件 `CLAUDE.md`。
- 已为三个子项目补充 Codex 规则文件：
  - `games/lol-helper/AGENTS.md`
  - `games/littledota-helper/AGENTS.md`
  - `games/lushi-cheater/AGENTS.md`

## 进行中

- 待确认各子项目的真实主线和可运行状态。

## 待办

- 确认 `games/littledota-helper/frontend/` 与 `games/littledota-helper/backend/frontend/` 哪个是主前端目录。
- 验证 `games/lushi-cheater/main_entry.py` 的导入路径和运行方式。
- 为 `games/lol-helper` 运行一次当前环境可承受的最小测试。
- 为 `games/littledota-helper` 后端补充明确的本地运行和验证说明。

## 阻塞

- 未确认当前机器是否已安装各子项目完整依赖。
- `lushi-cheater` 依赖游戏窗口、截图区域和本地图片路径，无法仅通过静态读取确认可运行。

## 最近验证

- 2026-06-22：读取根目录 README、AGENTS 和 `.gitignore`，确认文档写入正常。
- 2026-06-22：检查 `git status --short`，确认 `.codegraph/` 已被忽略。

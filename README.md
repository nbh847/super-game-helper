# super-game-helper

Playing with AI Instead of Hands.

## 项目定位

`super-game-helper` 是一个游戏 AI / 自动化助手实验仓库，用来沉淀不同游戏场景下的识别、决策、训练和操作模拟能力。

当前仓库按游戏拆分子项目，每个子项目独立维护自己的代码、依赖、数据和文档。

## 子项目

| 子项目 | 说明 | 主要技术 |
| --- | --- | --- |
| `games/lol-helper` | 英雄联盟极地大乱斗 AI 助手，目标是基于视觉识别、行为克隆和强化学习完成自动操作实验。 | Python, PyTorch, OpenCV, DQN |
| `games/littledota-helper` | 小冰冰传奇怀旧服助手，包含竞技场阵容破解后端和 Vue 前端。 | Python, Flask, SQLite, Vue, Vite |
| `games/lushi-cheater` | 炉石传说自动化脚本，基于截图匹配和鼠标键盘模拟执行固定流程。 | Python, PyAutoGUI, 图像匹配 |

## 目录结构

```text
.
├── games/
│   ├── lol-helper/
│   │   ├── backend/
│   │   ├── data/
│   │   └── docs/
│   ├── littledota-helper/
│   │   ├── backend/
│   │   └── frontend/
│   └── lushi-cheater/
│       ├── imgs/
│       ├── logic/
│       └── screen_shut_file/
├── LICENSE
└── README.md
```

## 快速入口

- LOL 助手文档：`games/lol-helper/docs/README.md`
- LOL 助手后端入口：`games/lol-helper/backend/main.py`
- LittleDota 后端入口：`games/littledota-helper/backend/main.py`
- LittleDota 前端入口：`games/littledota-helper/frontend/`
- 炉石脚本入口：`games/lushi-cheater/main_entry.py`

## 开发约定

- 根目录只放仓库级说明、许可证和通用配置。
- 新游戏助手放在 `games/<game-name>-helper/` 下。
- 子项目依赖、数据、文档和验证方式由子项目自己维护。
- 修改某个子项目时，优先阅读该子项目的 README 和 docs。
- 大型训练数据、模型文件、日志和临时截图不应提交到仓库。

## 当前状态

- `lol-helper`：文档和后端模块相对完整，已有测试与训练脚本。
- `littledota-helper`：后端接口和前端项目均已存在，需要确认主前端目录。
- `lushi-cheater`：早期脚本形态，存在硬编码路径和旧包名导入风险。

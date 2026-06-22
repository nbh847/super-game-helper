# AGENTS.md

本文件是 `super-game-helper` 仓库的 Codex 工作规则。所有在本仓库内执行的 Agent 都必须先读本文件，再修改项目。

## 沟通规则

- 默认使用中文回复。
- 每次回复先称呼用户为「主人」。
- 结论先行，理由随后。
- 不要夸赞、讨好或默认附和；发现问题直接指出。
- 需求不明确时，先给最合理方案；涉及高风险操作时必须先问。

## 仓库定位

`super-game-helper` 是游戏 AI / 自动化助手实验仓库，用于沉淀不同游戏场景下的识别、决策、训练和操作模拟能力。

当前按游戏拆分子项目：

- `games/lol-helper/`：英雄联盟极地大乱斗 AI 助手。
- `games/littledota-helper/`：小冰冰传奇怀旧服助手。
- `games/lushi-cheater/`：炉石传说自动化脚本。

## 目录约定

- 根目录只放仓库级说明、许可证、通用配置和 Agent 规则。
- 新游戏助手放在 `games/<game-name>-helper/` 下。
- 子项目自己的依赖、数据、文档、测试和运行方式放在子项目目录内。
- 大型训练数据、模型文件、日志、临时截图、索引目录不得提交。
- CodeGraph 本地索引目录 `.codegraph/` 必须保持忽略。

## 修改原则

- 只改用户请求直接相关的文件。
- 不做顺手重构、格式美化或无关清理。
- 匹配现有代码风格，即使风格不理想。
- 修改规范时先改文档，再按文档实践。
- 发现无关问题可以在最终回复中说明，不要自行修复。

## 红线操作

以下操作必须先获得用户明确确认：

- 删除文件或目录。
- git 回滚、reset、checkout 覆盖工作区。
- 修改 `.env`、密钥、token 或 CI/CD 配置。
- 数据库 schema 变更或数据迁移。
- 安装新的全局依赖或修改系统配置。
- 发布、部署、公开推送生产内容。

## 子项目入口

### `games/lol-helper/`

- 文档入口：`games/lol-helper/docs/README.md`
- 后端入口：`games/lol-helper/backend/main.py`
- 依赖文件：`games/lol-helper/backend/requirements.txt`
- 测试目录：`games/lol-helper/backend/tests/`
- 进度文档：`games/lol-helper/docs/progress.md`

### `games/littledota-helper/`

- 后端入口：`games/littledota-helper/backend/main.py`
- 后端依赖：`games/littledota-helper/backend/requirements.txt`
- 前端入口：`games/littledota-helper/frontend/`
- 前端依赖：`games/littledota-helper/frontend/package.json`

注意：当前还存在 `games/littledota-helper/backend/frontend/`，修改前需要确认主前端目录，避免重复维护。

### `games/lushi-cheater/`

- 脚本入口：`games/lushi-cheater/main_entry.py`
- 核心逻辑：`games/lushi-cheater/logic/`
- 图片素材：`games/lushi-cheater/imgs/`

注意：该子项目存在旧路径和旧包名导入风险，修改前需要先验证运行路径。

## 验证要求

- 修改代码后必须运行相关子项目的最小验证。
- 只改文档时，至少读取修改后的文件并检查 git 状态。
- 如果无法运行验证，必须在最终回复中说明原因。
- 不要通过注释报错代码或添加绕过标记来让验证通过。

常用验证参考：

```bash
# 根目录文档/配置检查
git status --short

# LOL helper
cd games/lol-helper/backend
python -m unittest discover tests

# LittleDota frontend
cd games/littledota-helper/frontend
npm run build
```

## 进度维护

- `games/lol-helper/docs/progress.md` 是 `lol-helper` 当前进度记录。
- 如果未来根目录新增 `ROADMAP.md`，则根目录级开发完成后需要同步更新。
- 只有已经实现并验证过的事项才能写入已完成。
- 未确认信息写「待确认」，不要猜。

# CLAUDE.md

本文件是 `super-game-helper` 仓库的 Claude Code 工作规则。Claude Code 在本仓库执行任务前必须先读取本文件。

## 基本沟通

- 默认使用中文。
- 每次回复先称呼用户为「主人」。
- 结论先行，再给理由。
- 不要夸赞、讨好或默认附和；发现问题直接指出。
- 需求不明确时，先给最合理方案；命中红线时必须先问。

## 仓库定位

`super-game-helper` 是游戏 AI / 自动化助手实验仓库，按游戏拆分子项目：

- `games/lol-helper/`：英雄联盟极地大乱斗 AI 助手。
- `games/littledota-helper/`：小冰冰传奇怀旧服助手。
- `games/lushi-cheater/`：炉石传说自动化脚本。

## 工作规则

- 只修改用户请求直接相关的文件。
- 不做顺手重构、格式美化或无关清理。
- 修改前先读相关子项目的 README、docs 和本目录下的规则文件。
- 修改规范时先改文档，再按文档实践。
- 大型训练数据、模型文件、日志、临时截图和 `.codegraph/` 不得提交。

## 红线操作

以下操作必须先获得用户明确确认：

- 删除文件或目录。
- git 回滚、reset、checkout 覆盖工作区。
- 修改 `.env`、密钥、token 或 CI/CD 配置。
- 数据库 schema 变更或数据迁移。
- 安装新的全局依赖或修改系统配置。
- 发布、部署、公开推送生产内容。

## 子项目入口

- LOL 文档：`games/lol-helper/docs/README.md`
- LOL 后端：`games/lol-helper/backend/main.py`
- LittleDota 后端：`games/littledota-helper/backend/main.py`
- LittleDota 前端：`games/littledota-helper/frontend/`
- 炉石脚本：`games/lushi-cheater/main_entry.py`

## 验证要求

- 修改代码后必须运行相关子项目的最小验证。
- 只改文档时，至少读取修改后的文件并检查 git 状态。
- 无法运行验证时，在最终回复中说明原因。

常用验证参考：

```bash
git status --short

cd games/lol-helper/backend
python -m unittest discover tests

cd games/littledota-helper/frontend
npm run build
```

## 进度维护

- 仓库级进度记录在 `ROADMAP.md`。
- `lol-helper` 细节进度记录在 `games/lol-helper/docs/progress.md`。
- 完成开发、修复、文档补齐或重要调研后，必须同步更新对应进度文件。
- 只有已经实现并验证过的事项才能写入已完成。

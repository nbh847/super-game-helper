# AGENTS.md

本文件是 `games/littledota-helper` 子项目的 Codex 工作规则。进入本子项目工作前，先读根目录 `AGENTS.md`，再读本文件。

## 项目定位

`littledota-helper` 是小冰冰传奇怀旧服助手，当前包含 Flask 后端、SQLite/Excel 数据和 Vue 前端。

## 关键入口

- 子项目 README：`README.md`
- 后端入口：`backend/main.py`
- 后端依赖：`backend/requirements.txt`
- 阵容逻辑：`backend/logic/xbb.py`
- 数据目录：`backend/data/`
- 主前端目录：`frontend/`
- 前端依赖：`frontend/package.json`

## 已知风险

- 当前存在两个前端目录：`frontend/` 和 `backend/frontend/`。修改前必须确认要改哪个目录；默认主线按根目录 README 记录为 `frontend/`。
- 后端逻辑依赖 `backend/data/arena.db`、`backend/data/arena_data.xlsx` 和 JSON 数据文件，改数据结构前必须先确认影响。

## 修改规则

- 不修改数据库 schema 或迁移数据，除非用户明确要求。
- 不提交临时导出的 Excel、数据库备份或构建产物。
- 后端接口保持现有输入输出结构，除非用户明确要求调整。
- 前端改动优先沿用现有 Vue/Vite 项目结构和组件风格。

## 验证

根据改动范围运行最小验证：

```bash
cd games/littledota-helper/frontend
npm run build
```

后端当前没有统一测试命令。修改后端时，至少用 Python 导入或 Flask 本地启动方式验证；如果依赖不完整，记录具体错误。

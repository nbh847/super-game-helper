# AGENTS.md

本文件是 `games/lol-helper` 子项目的 Codex 工作规则。进入本子项目工作前，先读根目录 `AGENTS.md`，再读本文件。

## 项目定位

`lol-helper` 是英雄联盟极地大乱斗 AI 助手实验项目，目标是基于视觉识别、行为克隆和强化学习完成自动操作能力验证。

## 关键入口

- 总览文档：`docs/README.md`
- 架构文档：`docs/architecture.md`
- 进度文档：`docs/progress.md`
- 后端入口：`backend/main.py`
- 依赖文件：`backend/requirements.txt`
- 测试目录：`backend/tests/`

## 修改规则

- 修改前先读相关 docs，不要只看代码猜设计。
- 保持 Python 3.9+ 兼容。
- 不要提交录像、训练数据、模型权重、日志或临时截图。
- 不要为了跑通测试禁用游戏操作、图像识别或输入模拟逻辑；需要跨平台兼容时用已有平台判断方式。
- 涉及外部 API、游戏协议、录像解析或 SDK 时，先查官方文档或可运行参考实现。

## 双环境约定

- 当前主开发机是 macOS，用于代码逻辑开发、文档维护、静态检查、mock 测试和不依赖游戏窗口/GPU 的轻量验证。
- 训练和真实游戏验证在 Windows 10 电脑执行，硬件为 RTX 5060 8GB 显卡。
- Windows 10 环境负责完整依赖安装、CUDA/GPU 训练、pydirectinput 输入模拟、真实游戏窗口截屏、端到端推理验证。
- macOS 上缺少训练依赖、CUDA、pydirectinput 或游戏窗口时，不视为项目失败；需要记录具体缺失项，并把对应验证标记为 Windows 10 必跑。
- 后续计划和最终回复中要区分「Mac 可验证」和「Windows 10 必验证」，不要把 Mac 的轻量验证等同于完整训练验证。

## 验证

优先运行最小相关测试。常用命令：

```bash
cd games/lol-helper/backend
python -m unittest discover tests
```

如果依赖不完整导致无法运行，记录具体缺失依赖或错误信息，不要猜。

## 进度维护

- 完成 `lol-helper` 相关开发、修复、文档补齐或重要调研后，更新 `docs/progress.md`。
- 只有已经实现并验证过的内容才能写入已完成。
- 未验证内容写入待办或待确认。

# Windows 环境配置指南

## 概述

本项目在 macOS 上开发，但训练任务需要在 Windows 10 上运行。本文档提供 Windows 环境的配置和运行指南。

---

## 系统要求

### Windows 系统
- Windows 10 或 Windows 11
- 64 位操作系统

### 硬件要求
- 显卡：NVIDIA RTX 5060 8GB（推荐）或兼容显卡
- 内存：16GB（推荐）
- 存储：至少 20GB 可用空间

### 软件要求
- Python 3.9 或 3.10
- CUDA 11.8 或 12.1（如果使用 GPU）
- Git（可选，用于克隆项目）

---

## 安装步骤

### 1. 安装 Python

1. 访问 [Python官网](https://www.python.org/downloads/)
2. 下载 Python 3.9 或 3.10 安装包
3. 运行安装程序，**务必勾选 "Add Python to PATH"**
4. 验证安装：
   ```cmd
   python --version
   pip --version
   ```

### 2. 克隆项目

如果使用 Git：
```cmd
git clone https://github.com/nbh847/super-game-helper.git
cd super-game-helper/games/lol-helper
```

或者直接下载项目文件夹到 `C:\projects\lol-helper`

### 3. 创建虚拟环境

```cmd
cd C:\projects\lol-helper
python -m venv venv
```

### 4. 激活虚拟环境

```cmd
venv\Scripts\activate
```

### 5. 安装依赖

**CPU 版本**（默认）：
```cmd
cd backend
pip install -r requirements.txt
```

**GPU 版本**（推荐，有 NVIDIA 显卡）：
```cmd
cd backend
# 先安装 PyTorch GPU 版本
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118
# 再安装其他依赖
pip install -r requirements.txt
```

### 6. 验证安装

```cmd
python -c "import torch; print(f'Torch: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import pydirectinput; print('pydirectinput: OK')"
```

---

## 跨平台兼容性说明

### pydirectinput（仅 Windows）

`pydirectinput` 是专门为游戏设计的输入库，**仅支持 Windows**。

**用途**：
- 游戏专用输入（DirectX 级别）
- 更低的延迟
- 更好的游戏兼容性

**替代方案（macOS/Linux）**：
- macOS/Linux 会自动回退到 `pyautogui`
- 代码已自动检测平台并选择合适的库

### 平台检测

项目使用以下方式检测操作系统：

```python
import platform

IS_WINDOWS = platform.system() == 'Windows'
IS_MACOS = platform.system() == 'Darwin'
IS_LINUX = platform.system() == 'Linux'
```

如果 `IS_WINDOWS` 为 True，会使用 `pydirectinput`；否则使用 `pyautogui`。

---

## 依赖版本说明

### NumPy 版本冲突

**问题**：
- torch 2.2.2 需要 NumPy 1.x
- opencv-python 4.9.0+ 需要 NumPy 2.x

**解决方案**：
- 限制 NumPy 版本为 `>=1.24.0,<2`
- 限制 OpenCV 版本为 `>=4.8.0,<4.9.0`

**验证**：
```cmd
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
# 应该输出: NumPy: 1.26.4（或其他1.x版本）
```

---

## Windows 特定配置

### 1. 管理员权限

游戏输入可能需要管理员权限。

**以管理员身份运行**：
- 右键点击 CMD 或 PowerShell
- 选择 "以管理员身份运行"
- 激活虚拟环境后运行程序

### 2. 防病毒软件

某些防病毒软件可能会误报 AI 助手为外挂。

**解决方案**：
- 将项目目录添加到白名单
- 或在运行时暂时禁用防病毒软件

### 3. 游戏窗口捕获

确保游戏以"窗口模式"或"无边框窗口"运行，便于窗口捕获。

---

## 常见问题

### Q1: pydirectinput 导入失败

**错误**：`ImportError: No module named 'pydirectinput'`

**解决**：
```cmd
pip install pydirectinput>=1.0.4
```

### Q2: CUDA 不可用

**错误**：`CUDA available: False`

**解决**：
1. 检查显卡驱动是否最新
2. 安装正确版本的 CUDA Toolkit
3. 重新安装 PyTorch GPU 版本：
   ```cmd
   pip uninstall torch torchvision
   pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118
   ```

### Q3: OpenCV 错误

**错误**：`cv2.error: OpenCV(4.x) ...`

**解决**：
```cmd
pip uninstall opencv-python
pip install opencv-python>=4.8.0,<4.9.0
```

### Q4: 权限错误

**错误**：`PermissionError: [Errno 13] Permission denied`

**解决**：
- 以管理员身份运行
- 或修改项目文件夹权限

---

## 测试基础设施

在 Windows 上运行测试：

```cmd
cd C:\projects\lol-helper
venv\Scripts\activate
python backend\test_infrastructure.py
```

**预期输出**：
```
============================================================
基础设施集成测试
============================================================

=== 测试路径管理 ===
  ✓ 项目根目录: C:\projects\lol-helper
  ✓ 数据目录: C:\projects\lol-helper\data
  ✓ 模型目录: C:\projects\lol-helper\backend\ai\models
  ✓ 日志目录: C:\projects\lol-helper\logs
✓ 所有目录创建成功

=== 测试日志系统 ===
✓ 日志系统正常

=== 测试控制线程 ===
  [Windows] 使用 pydirectinput 进行输入操作
  测试队列限制...
  测试基本指令...
✓ 控制线程正常

=== 测试输入模拟器 ===
  [Windows] 使用 pydirectinput 进行输入操作
  测试鼠标移动...
  测试点击...
  测试按键...
✓ 输入模拟器正常

=== 集成测试 ===
  测试路径和日志...
  测试控制线程和日志...
  测试输入模拟器和日志...
  执行一些操作...
✓ 集成测试正常

============================================================
✓ 所有测试通过！
============================================================
```

---

## 数据准备

### 从 macOS 传输数据

如果数据在 macOS 上生成，可以使用以下方式传输到 Windows：

1. **压缩传输**：
   ```bash
   # macOS 上
   tar -czf lol-data.tar.gz data/
   ```

2. **使用云存储**（百度网盘、OneDrive等）

3. **Git LFS**（如果项目使用 Git LFS）

### 直接在 Windows 上生成

训练脚本会在 Windows 上生成必要的目录和文件。

---

## 训练任务运行

### 激活虚拟环境
```cmd
cd C:\projects\lol-helper
venv\Scripts\activate
```

### 运行训练
```cmd
cd backend
python train_pretrain.py
```

### 监控训练
使用 TensorBoard 查看训练进度：
```cmd
tensorboard --logdir logs/runs
```

在浏览器打开：`http://localhost:6006`

---

## 性能优化

### GPU 加速

确保使用 GPU 版本的 PyTorch：
```cmd
python -c "import torch; print(torch.cuda.is_available())"
# 应该输出: True
```

### 批量大小调整

根据显存大小调整 `batch_size`：

- 8GB 显卡：`batch_size = 16`
- 12GB 显卡：`batch_size = 32`
- 16GB 显卡：`batch_size = 64`

---

## 日志和调试

### 日志位置

- 日志文件：`logs/lol_YYYYMMDD_HHMMSS.log`
- 截图：`logs/captures/`
- 输出：`logs/outputs/`

### 调试模式

修改 `backend/config/settings.py`：
```python
@dataclass
class DebugConfig:
    enabled: bool = True
    log_level: str = "DEBUG"  # 改为 DEBUG
    save_screenshots: bool = True
    show_detections: bool = True
```

---

## 安全和注意事项

1. **仅用于学习和研究**
   - 不要在正式对局中使用
   - 遵守游戏服务条款

2. **防检测**
   - 保持合理胜率（50-60%）
   - 模拟人类操作（随机延迟、APM控制）
   - 定期"休息"（暂停操作）

3. **数据隐私**
   - 不要上传个人数据
   - 保护游戏账号信息

---

## 联系和支持

- 项目仓库：https://github.com/nbh847/super-game-helper
- 文档目录：`games/lol-helper/docs/`
- 问题反馈：提交 Issue

---

**文档版本**: 1.0
**创建日期**: 2026-01-20
**适用系统**: Windows 10/11
**Python 版本**: 3.9/3.10

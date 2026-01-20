# 英雄联盟AI助手 - 开发进度

## 项目概览

- **项目名称**: 英雄联盟极地大乱斗AI助手
- **目标**: 全自动操作大乱斗模式
- **技术路线**: 行为克隆 + 强化学习
- **个人配置**: RTX5060 8GB + 16GB内存
- **开发语言**: Python 3.9+
- **主要框架**: PyTorch, Stable-Baselines3, YOLOv8

---

## 进度总览

### ✅ 已完成

#### 1. 调研阶段（2026-01-20）
- ✅ 网络调研完成
  - Grok AI 92%胜率案例分析
  - AlphaStar 技术路线参考
  - TLoL、LeagueAI 等开源项目调研
- ✅ 技术选型确定
  - 行为克隆 + 强化学习
  - YOLOv8n + PaddleOCR
  - PyAutoGUI 键鼠模拟
- ✅ 文档建立完成
  - README.md
  - design_proposal.md
  - architecture.md
  - anti_detection.md
  - api_reference.md
  - data_collection.md

#### 2. 阶段1：基础框架搭建（2026-01-20）
- ✅ 创建完整目录结构
- ✅ 创建配置文件（settings.py, hero_profiles.py）
- ✅ 创建核心模块骨架（3个）
- ✅ 创建AI预训练模块骨架（6个）
- ✅ 创建AI微调模块骨架（2个）
- ✅ 创建工具模块骨架（4个）
- ✅ 创建数据处理模块骨架（1个）
- ✅ 创建main.py
- ✅ 设置Python虚拟环境
- ✅ 更新.gitignore

#### 3. 阶段2：数据处理（2026-01-20）
- ✅ 实现录像解析器（replay_parser.py）
- ✅ 实现状态提取器（state_extractor.py）
- ✅ 实现动作提取器（action_extractor.py）
- ✅ 实现数据加载器（data_loader.py）
- ✅ 实现录像转换器（replay_converter.py）

### ⏳ 进行中
- 无

### ⏸️ 待开始
- 阶段3：模型训练V1
- 阶段4：实时游戏识别
- 阶段5：操作执行器
- 阶段6：集成与测试

---

## 详细进度

### 阶段1：基础框架搭建 ✅

#### 完成内容

- [x] 创建完整目录结构
  - `backend/core/` - 核心业务逻辑
  - `backend/ai/pretrain/` - AI预训练
  - `backend/ai/finetune/` - AI微调
  - `backend/utils/` - 工具模块
  - `backend/data/` - 数据处理
  - `backend/config/` - 配置文件
  - `logs/` - 日志目录

- [x] 创建配置文件
  - `backend/config/settings.py` - 主配置类
    - ModelConfig - 模型配置
    - GameConfig - 游戏配置
    - HeroConfig - 英雄配置
    - AIConfig - AI配置
    - AntiDetectionConfig - 防检测配置
    - PathConfig - 路径配置
    - VisionConfig - 视觉配置
    - DebugConfig - 调试配置
  
  - `backend/config/__init__.py` - 英雄配置
    - 15个常用英雄配置
    - 英雄类型分类
    - 技能信息
    - 打法风格

- [x] 创建核心模块骨架（3个）
  - `backend/core/game_state.py` - 游戏状态识别器
  - `backend/core/action_executor.py` - 操作执行器
  - `backend/core/ai_engine.py` - AI决策引擎

- [x] 创建AI预训练模块骨架（6个）
  - `backend/ai/pretrain/model.py` - 策略网络模型
  - `backend/ai/pretrain/trainer.py` - 训练器
  - `backend/ai/pretrain/replay_parser.py` - 录像解析器
  - `backend/ai/pretrain/state_extractor.py` - 状态提取器
  - `backend/ai/pretrain/action_extractor.py` - 动作提取器
  - `backend/ai/pretrain/data_loader.py` - 数据加载器

- [x] 创建AI微调模块骨架（2个）
  - `backend/ai/finetune/arena_env.py` - 大乱斗环境
  - `backend/ai/finetune/agent.py` - RL智能体

- [x] 创建工具模块骨架（4个）
  - `backend/utils/screen_capture.py` - 屏幕截取器
  - `backend/utils/image_recognition.py` - 图像识别器
  - `backend/utils/input_simulator.py` - 输入模拟器
  - `backend/utils/human_behavior.py` - 人类行为模拟器

- [x] 创建数据处理模块骨架（1个）
  - `backend/data/replay_converter.py` - 录像转换器

- [x] 创建主程序
  - `backend/main.py` - 主程序入口

- [x] 设置Python虚拟环境
  - 创建venv
  - 升级pip
  - 安装基础依赖

- [x] 更新.gitignore
  - 添加venv忽略
  - 添加数据文件忽略（*.rofl, *.h5, *.hdf5）
  - 添加模型文件忽略（*.pth, *.pt, *.pkl）
  - 添加日志忽略（*.log）
  - 添加截图忽略（*.png, *.jpg）

#### 代码统计

- **文件数**: 28个新文件
- **代码行数**: ~1118行
- **模块数**: 17个

#### Git提交

- **Commit**: 192eeb3
- **Message**: feat: 搭建英雄联盟AI助手基础框架
- **Date**: 2026-01-20

---

### 阶段2：数据处理 ✅

#### 完成内容

- [x] replay_parser.py - 录像解析器（~197行）
  - 解析.rofl文件结构
  - 读取文件头和版本信息
  - 读取和解析元数据（游戏ID、时长、模式、玩家、区域）
  - 使用zstd解压数据负载
  - 解析数据包（简化版本）
  - 提取关键帧
  - 提取玩家操作序列
  - 验证录像文件

- [x] state_extractor.py - 状态特征提取器（~228行）
  - 提取英雄状态（位置、血量、蓝量、等级、金币、技能冷却、KDA）
  - 提取敌方英雄状态（最多5个，包括位置、血量、距离）
  - 提取小兵状态（最多10个，包括己方和敌方小兵）
  - 提取防御塔状态（位置、血量、距离、敌我判断）
  - 状态归一化到[0, 1]范围
  - 欧氏距离计算

  **特征维度**:
  - 英雄特征: 13个
  - 敌方特征: 20个 (5敌人 × 4特征)
  - 小兵特征: 30个 (10小兵 × 3特征)
  - 防御塔特征: 5个
  - **总计: 68个特征**

- [x] action_extractor.py - 动作特征提取器（~246行）
  - 动作类型枚举（32种动作）
    - 0-7: 8方向移动
    - 8: 攻击
    - 9-12: Q/W/E/R技能
    - 13-14: D/F召唤师技能
    - 15: 治疗
    - 16: 回城
    - 17: 停止
    - 18-21: 技能组合（V2+）
  - 移动方向编码（8个方向，基于角度计算）
  - 动作编码和解码
  - 动作验证
  - one-hot编码转换
  - 动作名称映射

- [x] data_loader.py - 数据加载器（~260行）
  - 从.h5文件加载数据
  - 支持批量加载（目录中的多个.h5文件）
  - PyTorch Dataset实现
  - PyTorch DataLoader配置
  - 数据增强
    - 镜像翻转（水平）
    - 空间平移（随机小范围）
    - 高斯噪声（低强度）
  - 训练集/验证集划分
  - 示例数据集生成
  - Pin memory加速

- [x] replay_converter.py - 录像转换器（~331行）
  - 转换单个.rofl文件到.h5格式
  - 批量转换目录中的.rofl文件
  - 提取帧数据（跳帧采样）
  - 生成模拟游戏数据（简化版本）
    - 生成敌方英雄
    - 生成小兵
    - 生成防御塔
  - 提取状态和动作序列
  - HDF5格式存储（支持gzip压缩）
  - 保存元数据
  - 转换统计和日志
  - 输出文件验证

#### 功能特性

- ✅ 支持.rofl文件解析
- ✅ 支持68维状态特征提取
- ✅ 支持32种动作编码
- ✅ 支持HDF5格式存储
- ✅ 支持数据增强（镜像、平移、噪声）
- ✅ 支持批量处理
- ✅ 进度条显示（tqdm）
- ✅ 完整的错误处理
- ✅ 转换统计信息

#### 代码统计

- **新增代码**: ~1220行
- **修改文件**: 6个
- **新增功能**: 5个主要模块

#### Git提交

- **Commit**: cf3d59d
- **Message**: feat: 实现数据处理模块
- **Date**: 2026-01-20

---

## 已知问题

### 1. 依赖兼容性问题

**问题描述**:
- torch需要numpy 1.x
- opencv-python需要numpy 2.x
- 两者存在依赖冲突

**当前状态**: 部分解决

**解决方案**:
- [ ] 方案1: 使用opencv-python-headless（推荐）
- [ ] 方案2: 降级opencv到兼容版本
- [ ] 方案3: 等待numpy 2.x完全支持torch

**临时方案**:
- 先不安装opencv-python，使用其他库或延迟处理

### 2. .rofl文件解析

**问题描述**:
- 当前是简化版本
- 实际.rofl格式更复杂
- 完全解析需要大量时间

**当前状态**: 基础框架完成

**解决方案**:
- [ ] 方案1: 使用TLoL开源解析器
- [ ] 方案2: 改进当前解析器
- [ ] 方案3: 使用HuggingFace解码数据集（推荐）

**推荐方案**:
- 使用HuggingFace解码数据集，跳过.rofl解析
- 或使用TLoL的解析逻辑

---

## 下一步计划

### 立即行动
1. 继续阶段3：模型训练V1

### 阶段3：模型训练V1 ⏸️

#### 待实现

- [ ] model.py - 完善前向传播
  - CNN特征提取完整实现
  - LSTM序列处理完整实现
  - 策略头完整实现
  - 批量推理支持

- [ ] trainer.py - 实现训练逻辑
  - train_epoch方法实现
  - train方法实现（多epoch）
  - validate方法实现
  - save_model/load_model实现
  - TensorBoard日志

- [ ] 创建训练脚本
  - 数据加载
  - 模型初始化
  - 训练循环
  - 评估和保存

- [ ] 模型训练
  - 使用示例数据集训练
  - 监控训练过程
  - 保存最佳模型

- [ ] 模型评估
  - 计算准确率
  - 分析损失曲线
  - 可视化结果

#### 预期目标
- 模型能够学习基本模式
- 训练损失稳定下降
- 能够预测基本动作

---

## 资源需求

### 当前配置

**硬件**:
- 显卡: RTX5060 8GB
- 内存: 16GB

**软件**:
- Python: 3.9+
- PyTorch: 2.2.2 (CPU版本)
- 其他: numpy, pandas, h5py, tqdm等

### 预计资源占用

**训练阶段**:
- 显存: 2-3GB
- 内存: 4-6GB

**推理阶段**:
- 显存: 1.5-2.5GB
- 内存: 2-3GB

---

## 数据需求

### 目标

- 每个英雄: 10-20局大乱斗录像
- 总计: 150-300局录像
- 当前: 0局（示例数据集已创建）

### 数据来源计划

**推荐方案**: 混合使用
1. 自己录制（主要）
2. HuggingFace数据集（辅助）
3. TLoL数据集（测试）

---

## Git历史

### 提交记录

1. **765bbf5** - docs: 更新英雄联盟AI项目文档，添加调研结果和数据收集方案
2. **192eeb3** - feat: 搭建英雄联盟AI助手基础框架
3. **cf3d59d** - feat: 实现数据处理模块

### 分支信息

- **当前分支**: master
- **远程同步**: 已同步
- **远程仓库**: github.com:nbh847/super-game-helper.git

---

## 技术栈

### 已安装的依赖

**深度学习**:
- torch==2.2.2 (CPU版本)
- torchvision==0.17.2

**数据处理**:
- numpy==1.26.4
- pandas
- h5py
- zstandard

**图像处理**:
- opencv-python==4.13.0.90

**其他**:
- pyyaml
- tqdm
- pillow

**未安装**（待安装）:
- ultralytics (YOLOv8)
- paddleocr
- pyautogui
- pynput
- stable-baselines3
- gymnasium

---

## 项目结构

```
games/lol-helper/
├── backend/                      # 后端代码
│   ├── core/                    # 核心业务逻辑 ✅
│   ├── ai/                      # AI模块
│   │   ├── pretrain/           # 预训练 ✅
│   │   └── finetune/           # 微调（骨架）
│   ├── utils/                   # 工具模块（骨架）
│   ├── data/                    # 数据处理 ✅
│   ├── config/                  # 配置文件 ✅
│   ├── main.py                 # 主程序 ✅
│   └── requirements.txt        # 依赖列表 ✅
├── docs/                       # 文档 ✅
│   ├── README.md
│   ├── design_proposal.md
│   ├── architecture.md
│   ├── anti_detection.md
│   ├── api_reference.md
│   ├── data_collection.md
│   └── progress.md            # 本文件
├── logs/                       # 日志目录
└── venv/                       # Python虚拟环境
```

---

## 里程碑

### 已完成 ✅
- [x] 2026-01-20: 项目启动
- [x] 2026-01-20: 调研完成
- [x] 2026-01-20: 文档建立
- [x] 2026-01-20: 基础框架搭建完成
- [x] 2026-01-20: 数据处理模块完成

### 进行中 ⏳
- 无

### 待完成 ⏸️
- [ ] 模型训练V1
- [ ] 实时游戏识别
- [ ] 操作执行器
- [ ] 集成与测试
- [ ] V2+: 技能释放
- [ ] V3+: 技能组合
- [ ] V4+: 完整技能

---

## 备注

### 重要提醒
- 所有代码遵循dev-workflow规范
- 使用中文交互
- 使用venv虚拟环境
- 样式分类整理
- 功能完成需自检

### 遇到的问题
1. numpy版本兼容性
2. opencv-python依赖问题
3. .rofl文件解析复杂

### 解决思路
1. 优先使用已有数据集（TLoL/HuggingFace）
2. 延迟解决依赖问题
3. 逐步完善功能

---

## 联系信息

- 项目仓库: github.com:nbh847/super-game-helper
- 文档目录: games/lol-helper/docs/

---

**文档版本**: 1.0
**创建日期**: 2026-01-20
**最后更新**: 2026-01-20
**当前状态**: 阶段2完成，准备进入阶段3

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
英雄状态标注工具
用于标注英雄联盟大乱斗游戏中英雄的状态（移动/攻击/技能/受伤/死亡）
结合YOLO+OCR自动推断和人工确认
"""

import os
import sys
import json
import cv2
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk

backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from utils.paths import PROJECT_ROOT, DATA_DIR
from utils.logger import logger


class LabelConfig:
    """标注配置"""
    
    LABELS = {
        '1': {'name': '移动', 'color': '#00FF00', 'key': '1'},
        '2': {'name': '攻击', 'color': '#FF0000', 'key': '2'},
        '3': {'name': '技能', 'color': '#0000FF', 'key': '3'},
        '4': {'name': '受伤', 'color': '#FFFF00', 'key': '4'},
        '5': {'name': '死亡', 'color': '#808080', 'key': '5'},
    }
    
    LABEL_TO_ID = {v['name']: k for k, v in LABELS.items()}
    ID_TO_LABEL = LABELS
    
    DISPLAY_SIZE = (1200, 900)
    FRAME_INTERVAL = 3  # 秒（每3秒1帧）
    MAX_FRAMES = 300  # 覆盖15分钟（15分×20帧/分=300帧）
    
    DATA_DIR = DATA_DIR / 'hero_states'
    FRAMES_DIR = 'frames'
    LABELS_FILE = 'labels.json'
    PROGRESS_FILE = 'progress.json'
    VIDEO_STATUS_FILE = DATA_DIR / 'video_status.json'
    STRATEGY_VERSION = "v1.0"  # 标注策略版本


class FrameExtractor:
    """录像帧提取器"""
    
    def __init__(self, video_path, hero_name, video_manager=None):
        self.video_path = Path(video_path)
        self.hero_name = hero_name
        self.output_dir = LabelConfig.DATA_DIR / hero_name / LabelConfig.FRAMES_DIR
        self.video_manager = video_manager
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"初始化帧提取器: {video_path} -> {self.output_dir}")
    
    def extract_frames(self, interval=LabelConfig.FRAME_INTERVAL, 
                      max_frames=LabelConfig.MAX_FRAMES):
        """
        从视频中提取帧
        
        Args:
            interval: 提取间隔（秒）
            max_frames: 最大帧数
        
        Returns:
            frame_paths: 提取的帧路径列表
        """
        cap = cv2.VideoCapture(str(self.video_path))
        
        if not cap.isOpened():
            logger.error(f"无法打开视频: {self.video_path}")
            return []
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"视频信息: FPS={fps}, 总帧数={total_frames}")
        
        frame_paths = []
        frame_count = 0
        extracted_count = 0
        
        while extracted_count < max_frames:
            frame_num = frame_count * fps * interval
            
            if frame_num >= total_frames:
                break
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_path = self.output_dir / f"frame_{extracted_count:04d}.png"
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(str(frame_path))
            
            extracted_count += 1
            frame_count += 1
        
        cap.release()
        
        logger.info(f"成功提取 {len(frame_paths)} 帧")
        
        if self.video_manager:
            video_hash = self.video_manager.register_video(self.video_path, self.hero_name)
            self.video_manager.update_video_status(video_hash, 'labeling', total_frames=len(frame_paths))
        
        return frame_paths
    
    @staticmethod
    def extract_from_directory(video_dir, hero_name, video_manager=None):
        """
        从目录提取所有视频的帧
        
        Args:
            video_dir: 视频目录
            hero_name: 英雄名称
            video_manager: 视频状态管理器
        
        Returns:
            all_frame_paths: 所有帧路径列表
        """
        video_dir = Path(video_dir)
        all_frame_paths = []
        
        video_files = list(video_dir.glob("*.mp4")) + \
                      list(video_dir.glob("*.avi")) + \
                      list(video_dir.glob("*.mkv")) + \
                      list(video_dir.glob("*.webm"))
        
        for i, video_file in enumerate(video_files):
            logger.info(f"处理视频 {i+1}/{len(video_files)}: {video_file}")
            
            extractor = FrameExtractor(video_file, hero_name, video_manager)
            frame_paths = extractor.extract_frames()
            all_frame_paths.extend(frame_paths)
        
        logger.info(f"总共提取 {len(all_frame_paths)} 帧")
        
        return all_frame_paths


class VideoStatusManager:
    """视频标注状态管理器"""
    
    def __init__(self):
        self.status_file = LabelConfig.VIDEO_STATUS_FILE
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.videos = {}
        self._load_status()
    
    def _load_status(self):
        """加载视频状态"""
        if self.status_file.exists():
            with open(self.status_file, 'r', encoding='utf-8') as f:
                self.videos = json.load(f)
            logger.info(f"加载视频状态: {len(self.videos)} 个视频")
    
    def save_status(self):
        """保存视频状态"""
        with open(self.status_file, 'w', encoding='utf-8') as f:
            json.dump(self.videos, f, ensure_ascii=False, indent=2)
        logger.info("视频状态已保存")
    
    def _get_video_key(self, video_path):
        """获取视频的唯一标识（文件名）"""
        video_path = Path(video_path)
        return video_path.name
    
    def register_video(self, video_path, hero_name):
        """注册视频并返回视频ID"""
        video_key = self._get_video_key(video_path)
        
        if video_key not in self.videos:
            self.videos[video_key] = {
                'video_path': str(video_path),
                'hero_name': hero_name,
                'status': 'pending',
                'total_frames': 0,
                'labeled_frames': 0,
                'strategy_version': LabelConfig.STRATEGY_VERSION,
                'history': [],
                'created_at': datetime.now().isoformat(),
                'completed_at': None
            }
            self.save_status()
            logger.info(f"注册新视频: {video_key}")
        else:
            current_version = self.videos[video_key].get('strategy_version', 'unknown')
            if current_version != LabelConfig.STRATEGY_VERSION:
                logger.info(f"视频 {video_key} 策略版本已更新: {current_version} -> {LabelConfig.STRATEGY_VERSION}")
        
        return video_key
    
    def update_video_status(self, video_key, status, total_frames=None, labeled_frames=None):
        """更新视频状态"""
        if video_key in self.videos:
            old_status = self.videos[video_key]['status']
            self.videos[video_key]['status'] = status
            
            if total_frames is not None:
                self.videos[video_key]['total_frames'] = total_frames
            
            if labeled_frames is not None:
                self.videos[video_key]['labeled_frames'] = labeled_frames
            
            if status == 'completed' and old_status != 'completed':
                self.videos[video_key]['completed_at'] = datetime.now().isoformat()
                self.videos[video_key]['strategy_version'] = LabelConfig.STRATEGY_VERSION
                self._add_to_history(video_key, status)
            
            self.save_status()
    
    def _add_to_history(self, video_key, status):
        """添加到历史记录"""
        if video_key in self.videos:
            history_entry = {
                'status': status,
                'strategy_version': LabelConfig.STRATEGY_VERSION,
                'labeled_frames': self.videos[video_key]['labeled_frames'],
                'total_frames': self.videos[video_key]['total_frames'],
                'timestamp': datetime.now().isoformat()
            }
            
            if 'history' not in self.videos[video_key]:
                self.videos[video_key]['history'] = []
            
            self.videos[video_key]['history'].append(history_entry)
    
    def backup_labels(self, video_key, hero_name):
        """备份当前标注数据"""
        if video_key not in self.videos:
            return False
        
        video_info = self.videos[video_key]
        if video_info['status'] != 'completed':
            return False
        
        labels_dir = LabelConfig.DATA_DIR / hero_name
        labels_file = labels_dir / LabelConfig.LABELS_FILE
        
        if not labels_file.exists():
            return False
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = labels_dir / f"labels_backup_{timestamp}.json"
        
        import shutil
        shutil.copy2(labels_file, backup_file)
        
        logger.info(f"标注数据已备份: {backup_file}")
        return True
    
    def get_video_status(self, video_path):
        """获取视频状态"""
        video_key = self._get_video_key(video_path)
        if video_key in self.videos:
            return self.videos[video_key]
        return None
    
    def is_video_completed(self, video_path):
        """检查视频是否已标注完成"""
        status = self.get_video_status(video_path)
        return status and status['status'] == 'completed'
    
    def needs_relabel(self, video_path):
        """检查是否需要重新标注"""
        status = self.get_video_status(video_path)
        if not status:
            return False
        
        current_version = status.get('strategy_version', 'unknown')
        return current_version != LabelConfig.STRATEGY_VERSION
    
    def list_videos(self, hero_name=None, status=None):
        """列出视频"""
        videos = []
        
        for video_key, info in self.videos.items():
            if hero_name and info['hero_name'] != hero_name:
                continue
            
            if status and info['status'] != status:
                continue
            
            current_version = info.get('strategy_version', 'unknown')
            needs_relabel = current_version != LabelConfig.STRATEGY_VERSION
            
            videos.append({
                'key': video_key,
                'path': info['video_path'],
                'hero': info['hero_name'],
                'status': info['status'],
                'total': info['total_frames'],
                'labeled': info['labeled_frames'],
                'version': current_version,
                'needs_relabel': needs_relabel,
                'completed_at': info.get('completed_at', None),
                'history': info.get('history', [])
            })
        
        return sorted(videos, key=lambda x: x['completed_at'] or '', reverse=True)
    
    def print_status(self):
        """打印视频状态"""
        videos = self.list_videos()
        
        if not videos:
            print("暂无视频记录")
            return
        
        print("\n" + "="*80)
        print(f"视频标注状态 (当前策略版本: {LabelConfig.STRATEGY_VERSION})")
        print("="*80)
        
        for v in videos:
            status_icon = {
                'pending': '⏸️',
                'labeling': '🏷️',
                'completed': '✅'
            }.get(v['status'], '❓')
            
            progress = f"{v['labeled']}/{v['total']}" if v['total'] > 0 else "0/0"
            percent = int(v['labeled'] / v['total'] * 100) if v['total'] > 0 else 0
            
            version_info = f" [策略: {v['version']}"
            if v['needs_relabel']:
                version_info += f" ⚠️需要重新标注]"
            else:
                version_info += "]"
            
            print(f"{status_icon} {v['key']} ({v['hero']}){version_info}")
            print(f"   状态: {v['status']} | 进度: {progress} ({percent}%)")
            if v['completed_at']:
                print(f"   完成时间: {v['completed_at']}")
            
            if len(v['history']) > 1:
                print(f"   标注历史: {len(v['history'])} 次")
            
            print()
    
    def print_history(self, video_path=None, hero_name=None):
        """打印标注历史"""
        videos = self.list_videos(hero_name)
        
        if video_path:
            video_key = self._get_video_key(video_path)
            videos = [v for v in videos if v['key'] == video_key]
        
        if not videos:
            print("未找到视频记录")
            return
        
        for v in videos:
            print("\n" + "="*80)
            print(f"标注历史: {v['key']}")
            print("="*80)
            
            if not v['history']:
                print("暂无标注历史")
                continue
            
            for i, entry in enumerate(v['history'], 1):
                print(f"\n{i}. 策略版本: {entry.get('strategy_version', 'unknown')}")
                print(f"   状态: {entry['status']}")
                print(f"   进度: {entry['labeled_frames']}/{entry['total_frames']}")
                print(f"   时间: {entry['timestamp']}")
    
    def set_strategy_version(self, version):
        """更新策略版本"""
        LabelConfig.STRATEGY_VERSION = version
        logger.info(f"标注策略版本已更新为: {version}")


class AutoLabeler:
    """自动标注器（YOLO+OCR）"""
    
    def __init__(self):
        self.yolo_model = None
        self.ocr = None
        
        self._init_models()
    
    def _init_models(self):
        """初始化YOLO和OCR模型"""
        try:
            from ultralytics import YOLO
            yolo_path = Path(PROJECT_ROOT) / "yolov8n.pt"
            
            if yolo_path.exists():
                self.yolo_model = YOLO(str(yolo_path))
                logger.info("YOLOv8n模型加载成功")
            else:
                logger.warning(f"YOLO模型不存在: {yolo_path}")
        except Exception as e:
            logger.warning(f"YOLO模型加载失败: {e}")
        
        try:
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(use_angle_cls=True, lang='ch')
            logger.info("PaddleOCR模型加载成功")
        except Exception as e:
            logger.warning(f"PaddleOCR加载失败: {e}")
    
    def predict_label(self, frame_path):
        """
        基于YOLO+OCR自动推断标签
        
        Args:
            frame_path: 帧路径
        
        Returns:
            predicted_label: 预测的标签
            confidence: 置信度
        """
        frame = cv2.imread(frame_path)
        
        if frame is None:
            return LabelConfig.LABELS['1']['name'], 0.0
        
        try:
            if self.yolo_model is not None:
                results = self.yolo_model(frame)
                
                detected_objects = []
                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        detected_objects.append(cls)
                
                label = self._heuristic_rule(detected_objects, frame)
                return label, 0.7
        except Exception as e:
            logger.warning(f"自动推断失败: {e}")
        
        return LabelConfig.LABELS['1']['name'], 0.3
    
    def _heuristic_rule(self, detected_objects, frame):
        """
        基于检测结果启发式推断标签
        
        Args:
            detected_objects: 检测到的对象列表
            frame: 原始帧
        
        Returns:
            label: 推断的标签名称
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dark_ratio = np.sum(gray < 50) / gray.size
        
        if dark_ratio > 0.8:
            return LabelConfig.LABELS['5']['name']  # 死亡
        
        if len(detected_objects) > 5:
            return LabelConfig.LABELS['2']['name']  # 攻击
        
        return LabelConfig.LABELS['1']['name']  # 默认移动


class LabelManager:
    """标签数据管理器"""
    
    def __init__(self, hero_name, video_manager=None, video_key=None):
        self.hero_name = hero_name
        self.labels_dir = LabelConfig.DATA_DIR / hero_name
        self.labels_file = self.labels_dir / LabelConfig.LABELS_FILE
        self.progress_file = self.labels_dir / LabelConfig.PROGRESS_FILE
        self.video_manager = video_manager
        self.video_key = video_key
        
        self.labels = {}
        self.progress = {
            'current_index': 0,
            'total_frames': 0,
            'labeled_frames': 0,
            'last_updated': None
        }
        
        self._load_data()
    
    def _load_data(self):
        """加载已有数据"""
        if self.labels_file.exists():
            with open(self.labels_file, 'r', encoding='utf-8') as f:
                self.labels = json.load(f)
            logger.info(f"加载标签数据: {len(self.labels)} 条")
        
        if self.progress_file.exists():
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                self.progress = json.load(f)
            logger.info(f"加载进度: {self.progress}")
    
    def save_data(self):
        """保存数据"""
        self.progress['last_updated'] = datetime.now().isoformat()
        
        with open(self.labels_file, 'w', encoding='utf-8') as f:
            json.dump(self.labels, f, ensure_ascii=False, indent=2)
        
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(self.progress, f, ensure_ascii=False, indent=2)
        
        if self.video_manager and self.video_key:
            total = self.progress['total_frames']
            labeled = len(self.labels)
            
            if total > 0 and labeled >= total:
                self.video_manager.update_video_status(self.video_key, 'completed', labeled_frames=labeled)
                logger.info(f"视频标注完成: {labeled}/{total}")
            else:
                self.video_manager.update_video_status(self.video_key, 'labeling', labeled_frames=labeled)
        
        logger.info("数据已保存")
    
    def set_label(self, frame_name, label):
        """设置帧标签"""
        self.labels[frame_name] = label
        self.progress['labeled_frames'] = len(set(self.labels.values()))
    
    def get_label(self, frame_name):
        """获取帧标签"""
        return self.labels.get(frame_name, None)
    
    def update_progress(self, current_index, total_frames):
        """更新进度"""
        self.progress['current_index'] = current_index
        self.progress['total_frames'] = total_frames
    
    def get_progress(self):
        """获取进度"""
        return self.progress


class LabelToolGUI:
    """标注工具GUI"""
    
    def __init__(self, root, hero_name, frame_paths, video_manager=None, video_hash=None):
        self.root = root
        self.hero_name = hero_name
        self.frame_paths = frame_paths
        self.current_index = 0
        self.video_manager = video_manager
        self.video_hash = video_hash
        
        self.label_manager = LabelManager(hero_name, video_manager, video_hash)
        self.auto_labeler = AutoLabeler()
        
        self.current_image = None
        self.original_image = None
        self.predicted_label = None
        
        self.setup_ui()
        self.load_frame(0)
    
    def setup_ui(self):
        """设置UI"""
        self.root.title(f"英雄状态标注工具 - {self.hero_name}")
        self.root.geometry("1000x800")
        
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 顶部信息栏
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.info_label = ttk.Label(info_frame, text="", font=('Arial', 10))
        self.info_label.pack(side=tk.LEFT)
        
        # 图片显示区域
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.image_label = ttk.Label(image_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # 当前标签显示
        label_frame = ttk.Frame(main_frame)
        label_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.current_label_label = ttk.Label(
            label_frame, 
            text="当前标签: 未标注", 
            font=('Arial', 14, 'bold')
        )
        self.current_label_label.pack()
        
        self.predicted_label_label = ttk.Label(
            label_frame,
            text="",
            font=('Arial', 10),
            foreground='gray'
        )
        self.predicted_label_label.pack()
        
        # 快捷键提示
        hint_frame = ttk.Frame(main_frame)
        hint_frame.pack(fill=tk.X)
        
        hint_text = "快捷键: "
        for key, label_info in LabelConfig.LABELS.items():
            hint_text += f"[{key}]{label_info['name']} "
        
        hint_text += " | [←]上一帧 [→]下一帧 [S]保存 [Q]退出"
        
        hint_label = ttk.Label(hint_frame, text=hint_text, font=('Arial', 10))
        hint_label.pack(side=tk.LEFT)
        
        # 绑定快捷键
        self.root.bind('1', lambda e: self.set_label('1'))
        self.root.bind('2', lambda e: self.set_label('2'))
        self.root.bind('3', lambda e: self.set_label('3'))
        self.root.bind('4', lambda e: self.set_label('4'))
        self.root.bind('5', lambda e: self.set_label('5'))
        
        self.root.bind('Left', lambda e: self.previous_frame())
        self.root.bind('Right', lambda e: self.next_frame())
        self.root.bind('s', lambda e: self.save_data())
        self.root.bind('S', lambda e: self.save_data())
        self.root.bind('q', lambda e: self.exit_tool())
        self.root.bind('Q', lambda e: self.exit_tool())
        
        # 绑定窗口大小变化事件
        self.root.bind('<Configure>', self.on_window_resize)
        
        # 按钮栏
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(button_frame, text="上一帧", command=self.previous_frame).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="下一帧", command=self.next_frame).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="保存", command=self.save_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="退出", command=self.exit_tool).pack(side=tk.RIGHT, padx=5)
    
    def load_frame(self, index):
        """加载帧"""
        if 0 <= index < len(self.frame_paths):
            self.current_index = index
            frame_path = self.frame_paths[index]
            frame_name = Path(frame_path).name
            
            # 加载原始图片
            image = Image.open(frame_path)
            self.original_image = image
            
            # 等待窗口更新后再调整图片大小
            self.root.update_idletasks()
            self._resize_and_display_image()
            
            # 获取已有标签
            existing_label = self.label_manager.get_label(frame_name)

            if existing_label:
                label_id = LabelConfig.LABEL_TO_ID[existing_label]
                color = LabelConfig.LABELS[label_id]['color']
                self.current_label_label.configure(
                    text=f"当前标签: {existing_label}",
                    foreground=color
                )
            else:
                self.current_label_label.configure(
                    text="当前标签: 未标注",
                    foreground='black'
                )
            
            # 自动推断标签
            predicted, confidence = self.auto_labeler.predict_label(frame_path)
            self.predicted_label = predicted
            self.predicted_label_label.configure(
                text=f"AI建议: {predicted} (置信度: {confidence:.2f})"
            )
            
            # 更新信息
            progress = self.label_manager.get_progress()
            self.info_label.configure(
                text=f"帧: {index+1}/{len(self.frame_paths)} | "
                     f"已标注: {progress['labeled_frames']}/{progress['total_frames']}"
            )
            
            self.label_manager.update_progress(index, len(self.frame_paths))
    
    def _resize_and_display_image(self):
        """根据窗口大小调整并显示图片"""
        if not hasattr(self, 'original_image') or self.original_image is None:
            return
        
        # 获取图片显示区域的大小
        self.image_label.update_idletasks()
        label_width = self.image_label.winfo_width()
        label_height = self.image_label.winfo_height()
        
        # 如果窗口还未显示，使用默认大小
        if label_width <= 1 or label_height <= 1:
            label_width, label_height = LabelConfig.DISPLAY_SIZE
        
        # 计算保持宽高比的缩放
        img_width, img_height = self.original_image.size
        ratio = min(label_width / img_width, label_height / img_height)
        
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)
        
        # 缩放图片
        resized_image = self.original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 显示图片
        photo = ImageTk.PhotoImage(resized_image)
        self.image_label.configure(image=photo)
        self.image_label.image = photo
    
    def on_window_resize(self, event):
        """窗口大小改变时重新调整图片"""
        # 只在主窗口大小改变时重新调整图片
        if event.widget == self.root and hasattr(self, 'original_image'):
            self._resize_and_display_image()
    
    def set_label(self, label_id):
        """设置标签"""
        frame_name = Path(self.frame_paths[self.current_index]).name
        label_name = LabelConfig.LABELS[label_id]['name']
        
        self.label_manager.set_label(frame_name, label_name)
        
        color = LabelConfig.LABELS[label_id]['color']
        self.current_label_label.configure(
            text=f"当前标签: {label_name}",
            foreground=color
        )
        
        logger.info(f"帧 {frame_name} 标注为: {label_name}")
        
        # 自动跳到下一帧
        self.next_frame()
    
    def previous_frame(self):
        """上一帧"""
        if self.current_index > 0:
            self.load_frame(self.current_index - 1)
    
    def next_frame(self):
        """下一帧"""
        if self.current_index < len(self.frame_paths) - 1:
            self.load_frame(self.current_index + 1)
    
    def save_data(self):
        """保存数据"""
        self.label_manager.save_data()
        messagebox.showinfo("保存", "数据已保存!")
    
    def exit_tool(self):
        """退出工具"""
        if messagebox.askyesno("退出", "确定要退出吗?"):
            self.label_manager.save_data()
            self.root.destroy()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='英雄状态标注工具')
    parser.add_argument('--hero', type=str, required=True, help='英雄名称')
    parser.add_argument('--video', type=str, help='视频文件路径或目录')
    parser.add_argument('--frames-dir', type=str, help='已提取帧的目录')
    parser.add_argument('--status', action='store_true', help='显示视频标注状态')
    
    args = parser.parse_args()
    
    video_manager = VideoStatusManager()
    
    if args.status:
        video_manager.print_status()
        return
    
    frame_paths = []
    video_key = None
    
    if args.frames_dir:
        # 从已有帧目录加载
        frames_dir = Path(args.frames_dir)
        frame_paths = sorted(list(frames_dir.glob("*.png")))
        logger.info(f"从目录加载 {len(frame_paths)} 帧")
    
    elif args.video:
        # 从视频提取帧
        video_path = Path(args.video)
        
        if video_path.is_dir():
            frame_paths = FrameExtractor.extract_from_directory(video_path, args.hero, video_manager)
        else:
            # 检查视频是否已完成
            if video_manager.is_video_completed(video_path):
                print(f"\n⚠️ 视频 {video_path.name} 已标注完成！")
                status = video_manager.get_video_status(video_path)
                if status:
                    print(f"完成时间: {status.get('completed_at', '未知')}")
                    print(f"标注进度: {status.get('labeled_frames', 0)}/{status.get('total_frames', 0)}")
                
                response = input("\n是否继续重新标注? (y/N): ").strip().lower()
                if response != 'y':
                    print("已取消")
                    return
            
            extractor = FrameExtractor(video_path, args.hero, video_manager)
            frame_paths = extractor.extract_frames()
            video_key = video_manager._get_video_key(video_path)
    
    else:
        # 默认从数据目录加载
        frames_dir = LabelConfig.DATA_DIR / args.hero / LabelConfig.FRAMES_DIR
        if frames_dir.exists():
            frame_paths = sorted(list(frames_dir.glob("*.png")))
            logger.info(f"从默认目录加载 {len(frame_paths)} 帧")
        else:
            logger.error("未指定视频或帧目录，且默认目录不存在")
            sys.exit(1)
    
    if not frame_paths:
        logger.error("未找到任何帧")
        sys.exit(1)
    
    # 创建GUI
    root = tk.Tk()
    app = LabelToolGUI(root, args.hero, frame_paths, video_manager, video_key)
    root.mainloop()


if __name__ == '__main__':
    main()

"""
图像识别模块
使用YOLOv8n进行目标检测，PaddleOCR进行文本识别
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# YOLOv8
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[ImageRecognition] YOLOv8未安装，目标检测功能将被禁用")

# PaddleOCR
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    print("[ImageRecognition] PaddleOCR未安装，文本识别功能将被禁用")


class ImageRecognition:
    """
    图像识别器
    
    集成YOLOv8n目标检测和PaddleOCR文本识别
    """
    
    def __init__(self, 
                 yolo_model_path: str = "yolov8n.pt",
                 ocr_lang: str = "ch"):
        """
        初始化图像识别器
        
        Args:
            yolo_model_path: YOLOv8模型路径
            ocr_lang: OCR语言（'ch'中文，'en'英文）
        """
        self.yolo = None
        self.ocr = None
        
        # 初始化YOLOv8
        if YOLO_AVAILABLE:
            self.load_yolo_model(yolo_model_path)
        else:
            print("[ImageRecognition] 警告: YOLOv8未安装")
        
        # 初始化PaddleOCR
        if PADDLEOCR_AVAILABLE:
            self.load_ocr_model(lang=ocr_lang)
        else:
            print("[ImageRecognition] 警告: PaddleOCR未安装")
    
    def load_yolo_model(self, model_path: str = "yolov8n.pt"):
        """
        加载YOLOv8模型
        
        Args:
            model_path: 模型路径（'yolov8n.pt'使用预训练模型）
        """
        if not YOLO_AVAILABLE:
            return
        
        try:
            print(f"[ImageRecognition] 加载YOLOv8模型: {model_path}")
            self.yolo = YOLO(model_path)
            print(f"[ImageRecognition] YOLOv8加载成功")
        except Exception as e:
            print(f"[ImageRecognition] YOLOv8加载失败: {e}")
            self.yolo = None
    
    def load_ocr_model(self, lang: str = "ch"):
        """
        加载OCR模型
        
        Args:
            lang: OCR语言（'ch'中文，'en'英文）
        """
        if not PADDLEOCR_AVAILABLE:
            return
        
        try:
            print(f"[ImageRecognition] 加载PaddleOCR模型（语言: {lang}）")
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang=lang,
                show_log=False
            )
            print(f"[ImageRecognition] PaddleOCR加载成功")
        except Exception as e:
            print(f"[ImageRecognition] PaddleOCR加载失败: {e}")
            self.ocr = None
    
    def recognize_text(self, image: np.ndarray, 
                     region: Optional[Tuple[int, int, int, int]] = None) -> List[str]:
        """
        识别图像中的文本
        
        Args:
            image: 输入图像
            region: 文本区域 (left, top, width, height)，None表示全图
            
        Returns:
            识别的文本列表
        """
        if not PADDLEOCR_AVAILABLE or not self.ocr:
            return []
        
        # 裁剪区域
        if region is not None:
            left, top, width, height = region
            image = image[top:top+height, left:left+width]
        
        # OCR识别
        try:
            result = self.ocr.ocr(image, cls=True)
            
            # 提取文本
            texts = []
            if result and result[0]:
                for line in result[0]:
                    text = line[1][0]  # line[1]是(文本, 置信度)，line[1][0]是文本
                    texts.append(text)
            
            return texts
        except Exception as e:
            print(f"[ImageRecognition] OCR识别失败: {e}")
            return []
    
    def recognize_health(self, image: np.ndarray, 
                       health_bar_region: Tuple[int, int, int, int]) -> Optional[float]:
        """
        识别血量百分比
        
        Args:
            image: 输入图像
            health_bar_region: 血条区域 (left, top, width, height)
            
        Returns:
            血量百分比（0.0-1.0），识别失败返回None
        """
        if not PADDLEOCR_AVAILABLE or not self.ocr:
            return None
        
        # 裁剪血条区域
        left, top, width, height = health_bar_region
        health_bar = image[top:top+height, left:left+width]
        
        # OCR识别（可能识别"100%"、"80/100"等）
        texts = self.recognize_text(health_bar)
        
        if not texts:
            return None
        
        # 尝试解析血量
        for text in texts:
            # 尝试提取百分比
            if '%' in text:
                try:
                    value = text.replace('%', '').strip()
                    return float(value) / 100.0
                except ValueError:
                    continue
            
            # 尝试提取分数（如"800/1000"）
            if '/' in text:
                try:
                    parts = text.split('/')
                    current = float(parts[0])
                    total = float(parts[1])
                    return current / total if total > 0 else None
                except ValueError:
                    continue
        
        return None
    
    def recognize_gold(self, image: np.ndarray, 
                     gold_region: Tuple[int, int, int, int]) -> Optional[int]:
        """
        识别金币数量
        
        Args:
            image: 输入图像
            gold_region: 金币区域 (left, top, width, height)
            
        Returns:
            金币数量，识别失败返回None
        """
        if not PADDLEOCR_AVAILABLE or not self.ocr:
            return None
        
        # 裁剪金币区域
        left, top, width, height = gold_region
        gold_image = image[top:top+height, left:left+width]
        
        # OCR识别
        texts = self.recognize_text(gold_image)
        
        if not texts:
            return None
        
        # 尝试提取数字
        for text in texts:
            # 提取纯数字
            digits = ''.join(c for c in text if c.isdigit())
            if digits:
                try:
                    return int(digits)
                except ValueError:
                    continue
        
        return None
    
    def detect_objects(self, image: np.ndarray, 
                     classes: Optional[List[str]] = None,
                     conf_threshold: float = 0.5) -> List[Dict]:
        """
        检测图像中的物体
        
        Args:
            image: 输入图像
            classes: 过滤的类别列表，None表示检测所有类别
            conf_threshold: 置信度阈值
            
        Returns:
            检测结果列表，每个元素包含:
            {
                'bbox': (x1, y1, x2, y2),  # 边界框
                'class_id': int,            # 类别ID
                'class_name': str,          # 类别名称
                'confidence': float,         # 置信度
                'center': (x, y)           # 中心点
            }
        """
        if not YOLO_AVAILABLE or not self.yolo:
            return []
        
        try:
            # YOLO推理
            results = self.yolo(image, verbose=False)
            
            # 解析结果
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                
                for box in boxes:
                    # 获取置信度
                    conf = float(box.conf)
                    if conf < conf_threshold:
                        continue
                    
                    # 获取类别
                    cls_id = int(box.cls)
                    cls_name = self.yolo.names[cls_id]
                    
                    # 类别过滤
                    if classes is not None and cls_name not in classes:
                        continue
                    
                    # 获取边界框
                    bbox = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = bbox
                    
                    # 计算中心点
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'class_id': cls_id,
                        'class_name': cls_name,
                        'confidence': conf,
                        'center': (center_x, center_y)
                    })
            
            return detections
        except Exception as e:
            print(f"[ImageRecognition] 目标检测失败: {e}")
            return []
    
    def track_positions(self, detections: List[Dict]) -> List[Tuple[int, int]]:
        """
        获取所有检测物体的中心位置
        
        Args:
            detections: 检测结果列表
            
        Returns:
            中心位置列表 [(x, y), ...]
        """
        positions = []
        for det in detections:
            positions.append(det['center'])
        return positions
    
    def classify_hero(self, image: np.ndarray, 
                    hero_bbox: Tuple[int, int, int, int]) -> Optional[str]:
        """
        识别英雄类型（简化版本）
        
        注意：完整的英雄识别需要专门的分类模型
        这里只是一个示例实现
        
        Args:
            image: 输入图像
            hero_bbox: 英雄区域 (x1, y1, x2, y2)
            
        Returns:
            英雄名称，识别失败返回None
        """
        # 裁剪英雄区域
        x1, y1, x2, y2 = hero_bbox
        hero_image = image[y1:y2, x1:x2]
        
        # 简化实现：使用颜色特征分类
        # 实际项目中应该使用专门的分类模型
        
        # 提取主要颜色
        avg_color = np.mean(hero_image, axis=(0, 1))
        r, g, b = avg_color[2], avg_color[1], avg_color[0]
        
        # 简单的英雄分类（基于衣服颜色）
        # 这只是一个示例，实际需要更复杂的逻辑
        if r > 200 and g < 100 and b < 100:
            return "红色阵营英雄"
        elif r < 100 and g < 100 and b > 200:
            return "蓝色阵营英雄"
        else:
            return "未知英雄"
    
    def benchmark(self, image: np.ndarray, iterations: int = 100) -> dict:
        """
        性能测试
        
        Args:
            image: 测试图像
            iterations: 测试次数
            
        Returns:
            性能统计字典
        """
        print(f"[ImageRecognition] 开始性能测试，次数: {iterations}")
        
        results = {
            'yolo': None,
            'ocr': None
        }
        
        # 测试YOLO
        if YOLO_AVAILABLE and self.yolo:
            print("[ImageRecognition] 测试YOLO检测...")
            import time
            times = []
            for i in range(iterations):
                start = time.time()
                self.detect_objects(image, conf_threshold=0.5)
                times.append(time.time() - start)
            
            results['yolo'] = {
                'avg_time': np.mean(times) * 1000,  # 毫秒
                'min_time': np.min(times) * 1000,
                'max_time': np.max(times) * 1000,
                'fps': 1.0 / np.mean(times) if np.mean(times) > 0 else 0
            }
            print(f"  平均时间: {results['yolo']['avg_time']:.2f}ms")
            print(f"  FPS: {results['yolo']['fps']:.2f}")
        
        # 测试OCR
        if PADDLEOCR_AVAILABLE and self.ocr:
            print("[ImageRecognition] 测试OCR识别...")
            import time
            times = []
            for i in range(iterations):
                start = time.time()
                self.recognize_text(image)
                times.append(time.time() - start)
            
            results['ocr'] = {
                'avg_time': np.mean(times) * 1000,
                'min_time': np.min(times) * 1000,
                'max_time': np.max(times) * 1000,
                'fps': 1.0 / np.mean(times) if np.mean(times) > 0 else 0
            }
            print(f"  平均时间: {results['ocr']['avg_time']:.2f}ms")
            print(f"  FPS: {results['ocr']['fps']:.2f}")
        
        print(f"[ImageRecognition] 性能测试完成")
        return results


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("图像识别模块测试")
    print("=" * 60)
    
    # 创建测试图像
    test_image = np.zeros((180, 320, 3), dtype=np.uint8)
    test_image[:] = (50, 50, 50)  # 灰色背景
    
    # 创建识别器
    print("\n[1/3] 创建图像识别器...")
    recognizer = ImageRecognition()
    
    # 测试文本识别
    print("\n[2/3] 测试文本识别...")
    texts = recognizer.recognize_text(test_image)
    print(f"✓ 识别的文本: {texts}")
    
    # 测试目标检测
    print("\n[3/3] 测试目标检测...")
    detections = recognizer.detect_objects(test_image)
    print(f"✓ 检测到的物体数量: {len(detections)}")
    
    # 性能测试
    print("\n[4/4] 性能测试（10次）...")
    result = recognizer.benchmark(test_image, iterations=10)
    
    print("\n" + "=" * 60)
    print("✓ 所有测试完成！")
    print("=" * 60)

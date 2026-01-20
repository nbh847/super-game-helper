"""
日志系统模块
文件+控制台双输出，时间戳命名，支持多级别
"""

import os
import logging
from datetime import datetime
from pathlib import Path

from .paths import LOGS_DIR


class Logger:
    def __init__(self, name="lol_helper", level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self._setup_handlers()
    
    def _setup_handlers(self):
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = LOGS_DIR / f"lol_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        console_handler = logging.StreamHandler()
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"日志文件: {log_file}")
    
    def info(self, message):
        self.logger.info(message)
    
    def warning(self, message):
        self.logger.warning(message)
    
    def error(self, message):
        self.logger.error(message)
    
    def debug(self, message):
        self.logger.debug(message)


# 全局实例
logger = Logger()


if __name__ == "__main__":
    print("开始测试日志系统...")
    
    logger.info("这是一条info日志")
    logger.warning("这是一条warning日志")
    logger.error("这是一条error日志")
    logger.debug("这是一条debug日志")
    
    print("\n日志系统测试完成！")

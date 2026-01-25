#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据迁移脚本：将现有的英雄数据从扁平结构迁移到按视频组织的结构

旧结构：
data/hero_states/塞拉斯/
  ├── frames/
  │   ├── frame_0000.png
  │   └── frame_0001.png
  └── labels.json

新结构：
data/hero_states/塞拉斯/
  ├── record20260123-222005-塞拉斯-win.mp4/
  │   ├── frames/
  │   └── labels.json
  └── record20260126-080000-塞拉斯-win.mp4/
      ├── frames/
      └── labels.json
"""

import json
import shutil
from pathlib import Path
from datetime import datetime

def load_video_status():
    """加载video_status.json"""
    status_file = Path("data/hero_states/video_status.json")
    with open(status_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def backup_old_data(hero_dir):
    """备份旧数据"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = hero_dir.parent / f"{hero_dir.name}_backup_{timestamp}"
    if hero_dir.exists():
        shutil.copytree(hero_dir, backup_dir)
        print(f"  [OK] 已备份到: {backup_dir.name}")
    return backup_dir

def migrate_hero_data(hero_name, video_status):
    """迁移单个英雄的数据"""
    hero_dir = Path("data/hero_states") / hero_name
    old_frames_dir = hero_dir / "frames"
    old_labels_file = hero_dir / "labels.json"

    if not old_frames_dir.exists() or not old_labels_file.exists():
        print(f"  [SKIP] 跳过: {hero_name} (数据不完整)")
        return False

    # 找到该英雄的视频（包括completed和labeling）
    hero_videos = {
        video_name: video_info
        for video_name, video_info in video_status.items()
        if video_info['hero_name'] == hero_name and video_info['status'] in ['completed', 'labeling']
    }

    if not hero_videos:
        print(f"  [SKIP] 跳过: {hero_name} (没有找到已完成的视频)")
        return False

    # 读取旧标签
    with open(old_labels_file, 'r', encoding='utf-8') as f:
        old_labels = json.load(f)

    total_frames = len(old_labels)
    frames_per_video = total_frames // len(hero_videos)

    print(f"\n[MIGRATE] 迁移 {hero_name}:")
    print(f"   总帧数: {total_frames}")
    print(f"   视频数: {len(hero_videos)}")

    # 备份旧数据
    backup_old_data(hero_dir)

    # 为每个视频创建目录并分配帧
    frame_files = sorted(old_frames_dir.glob("frame_*.png"))

    for idx, (video_name, video_info) in enumerate(sorted(hero_videos.items())):
        start_idx = idx * frames_per_video
        end_idx = (idx + 1) * frames_per_video if idx < len(hero_videos) - 1 else len(frame_files)

        video_dir = hero_dir / video_name
        video_frames_dir = video_dir / "frames"
        video_labels_file = video_dir / "labels.json"

        # 创建目录
        video_frames_dir.mkdir(parents=True, exist_ok=True)

        # 迁移帧文件
        migrated_frames = 0
        total_to_migrate = len(frame_files[start_idx:end_idx])
        for frame_file in frame_files[start_idx:end_idx]:
            new_name = f"frame_{migrated_frames:04d}.png"
            shutil.copy2(frame_file, video_frames_dir / new_name)
            migrated_frames += 1
            
            # 显示进度
            if migrated_frames % 50 == 0 or migrated_frames == total_to_migrate:
                print(f"     进度: {migrated_frames}/{total_to_migrate} 帧", flush=True)

        # 创建新的labels.json
        video_labels = {}
        for frame_idx in range(migrated_frames):
            old_frame_name = f"frame_{start_idx + frame_idx:04d}.png"
            new_frame_name = f"frame_{frame_idx:04d}.png"

            if old_frame_name in old_labels:
                video_labels[new_frame_name] = old_labels[old_frame_name]

        # 保存labels.json
        with open(video_labels_file, 'w', encoding='utf-8') as f:
            json.dump(video_labels, f, ensure_ascii=False, indent=2)

        print(f"   [OK] {video_name}: {migrated_frames} 帧")

    # 重命名旧的frames和labels.json为old_*
    shutil.move(old_frames_dir, hero_dir / "old_frames")
    shutil.move(old_labels_file, hero_dir / "old_labels.json")

    print(f"   [OK] 迁移完成！旧数据已重命名为 old_*")
    return True

def main():
    print("=" * 80)
    print("数据迁移脚本")
    print("=" * 80)

    # 加载视频状态
    print("\n[LOAD] 加载视频状态...")
    video_status = load_video_status()
    print(f"   找到 {len(video_status)} 个视频记录")

    # 获取所有已完成的英雄（包括completed和labeling）
    completed_heroes = set()
    for video_info in video_status.values():
        if video_info['status'] in ['completed', 'labeling']:
            completed_heroes.add(video_info['hero_name'])
    
    print(f"   已完成的英雄: {len(completed_heroes)} 个")

    # 迁移每个英雄的数据
    success_count = 0
    for hero_name in sorted(completed_heroes):
        if migrate_hero_data(hero_name, video_status):
            success_count += 1

    print("\n" + "=" * 80)
    print(f"迁移完成: {success_count}/{len(completed_heroes)} 个英雄")
    print("=" * 80)
    print("\n提示：")
    print("  1. 检查迁移后的数据")
    print("  2. 确认无误后，可以删除 old_* 目录和备份")
    print("  3. 如果出现问题，可以恢复备份")

if __name__ == '__main__':
    main()

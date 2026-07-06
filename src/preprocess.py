"""
preprocess.py
Extract frames from all videos ONCE and save as images.
Run this once before training — never again.

Before: training reads video files every batch (slow)
After:  training reads pre-saved images (fast)
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import json

def extract_and_save_frames(
    video_dir,
    output_dir,
    json_path,
    num_frames=8
):
    """
    For each video in the dataset:
    1. Read the video file
    2. Extract 8 evenly spaced frames
    3. Save each frame as a .jpg file
    4. Create an index file mapping video_id to frame paths
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset to know which videos we need
    with open(json_path) as f:
        data = json.load(f)
    
    # Get unique video IDs that have matching files
    video_ids = []
    for item in data:
        video_path = os.path.join(video_dir, f"{item['id']}.mp4")
        if os.path.exists(video_path):
            video_ids.append(item['id'])
    
    # Remove duplicates
    video_ids = list(set(video_ids))
    print(f"Videos to process: {len(video_ids)}")
    
    # Track which videos were processed successfully
    processed = {}
    failed    = []
    
    for video_id in tqdm(video_ids, desc="Extracting frames"):
        video_path  = os.path.join(video_dir, f"{video_id}.mp4")
        frames_dir  = os.path.join(output_dir, video_id)
        
        # Skip if already extracted
        if os.path.exists(frames_dir):
            frame_files = [
                f for f in os.listdir(frames_dir)
                if f.endswith('.jpg')
            ]
            if len(frame_files) == num_frames:
                processed[video_id] = [
                    os.path.join(frames_dir, f"frame_{i}.jpg")
                    for i in range(num_frames)
                ]
                continue
        
        os.makedirs(frames_dir, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            failed.append(video_id)
            continue
        
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0:
            cap.release()
            failed.append(video_id)
            continue
        
        # Sample evenly spaced frames
        indices = np.linspace(
            0, total - 1, num_frames, dtype=int
        )
        
        frame_paths = []
        success = True
        
        for i, idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                # Use blank frame if read fails
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
            
            # Resize to 224x224 — save storage and loading time
            frame = cv2.resize(frame, (224, 224))
            
            # Save as JPEG (fast to read, reasonable quality)
            frame_path = os.path.join(
                frames_dir, f"frame_{i}.jpg"
            )
            cv2.imwrite(frame_path, frame, 
                       [cv2.IMWRITE_JPEG_QUALITY, 95])
            frame_paths.append(frame_path)
        
        cap.release()
        
        if success:
            processed[video_id] = frame_paths
    
    # Save index file
    index_path = os.path.join(output_dir, 'frame_index.json')
    with open(index_path, 'w') as f:
        json.dump(processed, f)
    
    print(f"\n✅ Done!")
    print(f"  Processed: {len(processed)} videos")
    print(f"  Failed:    {len(failed)} videos")
    print(f"  Index saved to: {index_path}")
    
    return processed


if __name__ == "__main__":
    extract_and_save_frames(
        video_dir  = 'videos/Soccer',
        output_dir = 'data/frames',
        json_path  = 'data/SportU_Video_mc.json',
        num_frames = 8
    )

# dataset.py
# Loads SPORTU soccer video data for training.

import json
import os
import cv2
from httpx import options
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# How to preprocess each video frame
# These values match what ResNet was trained on (ImageNet)
FRAME_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

# Convert answer letter to number
# A=0, B=1, C=2, D=3
ANSWER_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3}


def extract_frames(video_path, num_frames=8):
    """
    Extract 8 evenly spaced frames from a video.
    
    Why 8 frames?
    Enough to understand what's happening without
    being too slow to process.
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        # Return blank frames if video can't open
        blank = Image.fromarray(
            np.zeros((224, 224, 3), dtype=np.uint8)
        )
        return [blank] * num_frames
    
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        blank = Image.fromarray(
            np.zeros((224, 224, 3), dtype=np.uint8)
        )
        return [blank] * num_frames
    
    # Pick evenly spaced frame positions
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx) #open video
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
        else:
            blank = Image.fromarray(
                np.zeros((224, 224, 3), dtype=np.uint8)
            )
            frames.append(blank)
    
    cap.release()
    return frames


def preprocess_frames(frames):
    """
    Apply transforms to frames and stack into one tensor.
    Output shape: [8, 3, 224, 224]
    """
    tensors = [FRAME_TRANSFORM(f) for f in frames]
    return torch.stack(tensors)


class SPORTUDataset(Dataset):
    """
    PyTorch Dataset for SPORTU Soccer Video QA.
    
    Each example has:
    - Video frames tensor [8, 3, 224, 224]
    - Question + options as text
    - Correct answer as number (0,1,2,3)
    """
    
    def __init__(self, data, video_dir, processor, num_frames=8):
        """
        data:       list of examples from JSON
        video_dir:  folder containing .mp4 files
        processor:  CLIP processor for text
        num_frames: frames to sample per video
        """
        # Only keep examples that have matching video files
        self.data = []
        for item in data:
            video_path = os.path.join(
                video_dir, f"{item['id']}.mp4"
            )
            if os.path.exists(video_path):
                self.data.append(item)
        
        self.video_dir  = video_dir
        self.processor  = processor
        self.num_frames = num_frames
        
        print(f"Dataset ready: {len(self.data)} examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        #Load video
        video_path = os.path.join(
            self.video_dir, f"{item['id']}.mp4"
        )
        frames = extract_frames(video_path, self.num_frames)
        frames = preprocess_frames(frames)
        # Shape: [8, 3, 224, 224]
        
        #Process text
        question = item['question']
        options  = item['options']
        
        # Create "Question: Answer: " for each option
        available_keys = list(options.keys())

        # Always ensure we have exactly 4 options
        # If missing, duplicate the last available option
        while len(available_keys) < 4:
            available_keys.append(available_keys[-1])

        texts = [
            f"Question: {question} Answer: {options[available_keys[i]]}"
            for i in range(4)
        ]
        
        encoded = self.processor(
            text           = texts,
            return_tensors = "pt",
            padding        = 'max_length',
            truncation     = True,
            max_length     = 77
        )
        # Shape: [4, 77]
        
        # Get label
        label = ANSWER_MAP.get(item['answer'], 0)
        
        return {
            'frames'        : frames,
            'input_ids'     : encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'label'         : torch.tensor(label, dtype=torch.long),
            'video_id'      : item['id']
        }
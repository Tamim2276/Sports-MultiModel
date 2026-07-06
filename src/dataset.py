"""
dataset.py
Fast dataset that reads pre-extracted frames
instead of loading videos every batch.

Speed improvement: 15s/batch → <1s/batch
"""

import json
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Frame preprocessing
FRAME_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
    # No resize needed — already 224x224 from preprocess.py
])

ANSWER_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3}


class SPORTUDataset(Dataset):

    def __init__(self, data, frames_dir,
                 processor, num_frames=8):
        """
        data:       list of examples from JSON
        frames_dir: folder containing pre-extracted frames
        processor:  CLIP processor for text
        num_frames: frames per video (must match preprocess.py)
        """
        # Load frame index
        index_path = os.path.join(
            frames_dir, 'frame_index.json'
        )
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"Frame index not found at {index_path}\n"
                f"Run: python src/preprocess.py first!"
            )
        
        with open(index_path) as f:
            self.frame_index = json.load(f)
        
        # Only keep examples with pre-extracted frames
        self.data = []
        for item in data:
            if item['id'] in self.frame_index:
                self.data.append(item)
        
        self.frames_dir = frames_dir
        self.processor  = processor
        self.num_frames = num_frames
        
        print(f"Dataset ready: {len(self.data)} examples")
        print(f"Using pre-extracted frames from: {frames_dir}")
    
    def __len__(self):
        return len(self.data)
    
    def load_frames(self, video_id):
        # Load pre-saved JPEG frames for a video.
        frame_paths = self.frame_index[video_id]
        frames = []
        
        for path in frame_paths:
            try:
                img = Image.open(path).convert('RGB')
                frames.append(FRAME_TRANSFORM(img))
            except Exception:
                # Blank frame if loading fails
                frames.append(torch.zeros(3, 224, 224))
        
        # Stack into tensor [num_frames, 3, 224, 224]
        return torch.stack(frames)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        frames = self.load_frames(item['id'])
        # Shape: [8, 3, 224, 224]
        
        # Process text
        question = item['question']
        options  = item['options']
        
        available_keys = list(options.keys())
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
        
        #Label
        label = ANSWER_MAP.get(item['answer'], 0)
        
        return {
            'frames'        : frames,
            'input_ids'     : encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'label'         : torch.tensor(
                label, dtype=torch.long
            ),
            'video_id'      : item['id']
        }
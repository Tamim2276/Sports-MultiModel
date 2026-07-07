import json
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Define transforms outside the class
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

ANSWER_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

class SPORTUDataset(Dataset):
    def __init__(self, data, frames_dir, processor, num_frames=8, transform=None):
        index_path = os.path.join(frames_dir, 'frame_index.json')
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Frame index not found at {index_path}")
        
        with open(index_path) as f:
            self.frame_index = json.load(f)
        
        self.data = [item for item in data if item['id'] in self.frame_index]
        self.frames_dir = frames_dir
        self.processor = processor
        self.num_frames = num_frames
        self.transform = transform or EVAL_TRANSFORM # Default to EVAL if None provided
        
        print(f"Dataset ready: {len(self.data)} examples")

    def __len__(self):
        return len(self.data)
    
    def load_frames(self, video_id):
        frame_paths = self.frame_index[video_id]
        frames = []
        
        for path in frame_paths:
            try:
                img = Image.open(path).convert('RGB')
                frames.append(self.transform(img))
            except Exception:
                frames.append(torch.zeros(3, 224, 224))
        
        return torch.stack(frames)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        frames = self.load_frames(item['id'])
        
        question = item['question']
        options = item['options']
        all_keys = ['A', 'B', 'C', 'D']
        texts = [f"Question: {question} Answer: {options.get(key, 'N/A')}" for key in all_keys]
        
        encoded = self.processor(
            texts,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=128 
        )
        
        label = ANSWER_MAP[item['answer']]
        
        return {
            'frames': frames,
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'label': torch.tensor(label, dtype=torch.long),
            'video_id': item['id']
        }
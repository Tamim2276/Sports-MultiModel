import os
import json

# Check data files
data_path = 'D:/sports_thesis/data/SportU_Video_mc.json'
if os.path.exists(data_path):
    with open(data_path) as f:
        data = json.load(f)
    print(f' Annotation file found: {len(data)} examples')
else:
    print(' Annotation file not found — please download it first')

# Check videos folder
videos_path = 'D:/sports_thesis/videos'
videos = os.listdir(videos_path)
print(f' Videos folder has {len(videos)} items')
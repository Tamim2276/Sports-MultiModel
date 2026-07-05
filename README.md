# 🏆 Sports Video Question Answering — Multimodal Deep Learning

A multimodal deep learning system that watches football video clips and answers multiple-choice questions about them. Built with PyTorch, ResNet-18, and CLIP.

---

## 🧠 What This Model Does

This model solves **Video Question Answering (VQA)** for sports — specifically soccer. Given a short video clip and a multiple-choice question, it picks the correct answer by understanding both the visual content of the video and the meaning of the text.

### Example

```
Video:    A soccer clip showing players in black & blue jerseys
Question: What main color jersey is the offensive team wearing?

Options:
  A. red & white
  B. blue & red
  C. black & red
  D. black & blue  ✅ ← model picks this

```

### How It Works — Three Components

```
Video Frames ──→ [VideoEncoder]  ──→ 256-dim visual features
                                            ↓
                                    [Fusion Module] ──→ 4 scores ──→ Answer
                                            ↑
Text Options ──→ [TextEncoder]   ──→ 256-dim text features (×4)
```

**VideoEncoder** — Uses a pretrained ResNet-18 to extract visual features from 8 evenly sampled frames, then averages them into one video-level representation.

**TextEncoder** — Uses a pretrained CLIP model to convert each question+answer option pair into a semantic text vector.

**Fusion Module** — Concatenates video and text features for each option and scores them. The option with the highest score is the predicted answer.

### Dataset

Built on [SPORTU](https://github.com/chili-lab/SPORTU) — an ICLR 2025 benchmark for sports video understanding.

| Split      | Examples  |
| ---------- | --------- |
| Train      | 2,778     |
| Validation | 595       |
| Test       | 596       |
| **Total**  | **3,969** |

---

## 🖥️ Requirements

### Hardware

- CPU (required) — training runs on CPU by default
- GPU (optional) — Intel Arc / NVIDIA CUDA supported

### Software

- Windows 10/11 or Linux
- Python 3.10
- Miniconda or Anaconda

---

## ⚙️ Installation

### Step 1 — Clone the Repository

```bash
https://github.com/Tamim2276/Sports-MultiModel.git
cd sports_thesis
```

### Step 2 — Install Miniconda

Download from: https://docs.conda.io/en/latest/miniconda.html

Install it — on Windows, tick **"Add to PATH"** during installation.

### Step 3 — Create the Conda Environment

```bash
conda create -n sports_thesis python=3.10
conda activate sports_thesis
```

You should see `(sports_thesis)` at the start of your terminal line.

### Step 4 — Install PyTorch

```bash
pip install torch torchvision torchaudio
```

### Step 5 — Install All Other Dependencies

```bash
pip install transformers opencv-python pillow tqdm matplotlib jupyter ipykernel scikit-learn
```

### Step 6 — Set HuggingFace Cache Location (Optional but Recommended)

Prevents large model files from filling up your system drive:

```bash
# Windows
setx HF_HOME "D:\huggingface_cache"

# Linux/Mac
export HF_HOME="/path/to/large/drive/huggingface_cache"
```

Close and reopen your terminal after running this.

### Step 7 — Verify Installation

```bash
python -c "import torch; import cv2; import transformers; print('All good!')"
```

Expected output:

```
All good!
```

---

## 📦 Download the Dataset

### Step 1 — Download Annotation Files

Go to https://github.com/chili-lab/SPORTU and download the repository as a ZIP.

Copy these files into the `data/` folder:

```
data/
└── SportU_Video_mc.json
```

### Step 2 — Download Soccer Videos

Open this Google Drive link:

```
https://drive.google.com/drive/folders/1nvA8gqF32lrhqzhbJ2r39-TwwW5tEvsu
```

Download the **Soccer** folder and extract it into `videos/`:

```
videos/
└── Soccer/
    ├── soccer_1.mp4
    ├── soccer_2.mp4
    └── ... (506 clips)
```

---

## 📁 Project Structure

After setup, your folder should look like this:

```
sports_thesis/
│
├── data/
│   └── SportU_Video_mc.json     ← question & answer annotations
│
├── videos/
│   └── Soccer/
│       ├── soccer_1.mp4
│       └── ...                  ← 506 soccer video clips
│
├── notebooks/
│   ├── 00_gpu_test.ipynb        ← verify setup
│   └── 01_data_exploration.ipynb← explore the dataset
│
├── src/
│   ├── __init__.py
│   ├── dataset.py               ← data loading and preprocessing
│   ├── model.py                 ← VideoEncoder, TextEncoder, Fusion
│   └── train.py                 ← training and evaluation loops
│
├── outputs/
│   ├── best_model.pth           ← saved model weights (after training)
│   ├── training_curves.png      ← loss and accuracy plots
│   └── results.json             ← final accuracy numbers
│
├── main.py                      ← entry point — run this to train
└── README.md
```

---

## 🚀 How to Run

### Activate Environment First

Always activate your environment before running anything:

```bash
conda activate sports_thesis
cd sports_thesis
```

### Train the Model

```bash
python main.py
```

Expected output:

```
Using device: cpu
Dataset ready: 3969 examples
Train:2778 Val:595 Test:596
Trainable parameters: 427,009

========================================
Starting Training
========================================

Epoch 1/5
Training: 100%|████████| 695/695
  Train Loss: 1.3821 | Acc: 28.45%
  Val   Loss: 1.3654 | Acc: 31.20%
  ✅ Best model saved!

Epoch 2/5
...
```

Training takes approximately:

- CPU: 2–4 hours for 5 epochs
- GPU: 30–60 minutes for 5 epochs

### Change Training Settings

Open `main.py` and edit the config section at the top:

```python
BATCH_SIZE  = 4     # increase if you have more RAM
NUM_EPOCHS  = 5     # more epochs = longer training
LR          = 1e-4  # learning rate
FEATURE_DIM = 256   # feature vector size
NUM_FRAMES  = 8     # frames sampled per video
```

---

## ✅ How to Check if Everything is Working

### Check 1 — Libraries

```bash
python -c "import torch; import cv2; import transformers; print('All good!')"
```

✅ Expected: `All good!`

### Check 2 — Data Files

```bash
python -c "
import json, os
with open('data/SportU_Video_mc.json') as f:
    data = json.load(f)
print(f'Questions loaded: {len(data)}')
videos = os.listdir('videos/Soccer')
print(f'Videos found: {len(videos)}')
"
```

✅ Expected:

```
Questions loaded: 10973
Videos found: 506
```

### Check 3 — Model Builds Without Errors

```bash
python -c "
from src.model import SportsMultimodalModel
import torch
model = SportsMultimodalModel()
total = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Model built successfully!')
print(f'Trainable parameters: {total:,}')
"
```

✅ Expected:

```
Model built successfully!
Trainable parameters: 427,009
```

### Check 4 — One Forward Pass Works

```bash
python -c "
import torch
from src.model import SportsMultimodalModel

model = SportsMultimodalModel()
model.eval()

# Fake batch of 2 videos
frames         = torch.randn(2, 8, 3, 224, 224)
input_ids      = torch.randint(0, 1000, (2, 4, 77))
attention_mask = torch.ones(2, 4, 77, dtype=torch.long)

with torch.no_grad():
    scores = model(frames, input_ids, attention_mask)

print(f'Forward pass works!')
print(f'Output shape: {scores.shape}')
print(f'Scores: {scores}')
"
```

✅ Expected:

```
Forward pass works!
Output shape: torch.Size([2, 4])
Scores: tensor([[...]])
```

### Check 5 — After Training, Check Results

```bash
python -c "
import json
with open('outputs/results.json') as f:
    r = json.load(f)
print(f'Best Val Accuracy : {r[\"best_val_accuracy\"]:.2f}%')
print(f'Test Accuracy     : {r[\"test_accuracy\"]:.2f}%')
"
```

✅ Expected (anything above 25% means model is learning):

```
Best Val Accuracy : 38.50%
Test Accuracy     : 37.20%
```

---

## 📊 Understanding the Results

Random chance = 25% (4 options, pick one randomly).

| Accuracy | Meaning                                |
| -------- | -------------------------------------- |
| ~25%     | Model is not learning — same as random |
| 30–40%   | Model is learning basic patterns       |
| 40–55%   | Good performance for this architecture |
| 55%+     | Strong performance                     |

The training curves plot is saved to `outputs/training_curves.png` — check it to see if your model is improving over epochs.

---

## 🔧 Troubleshooting

**`ModuleNotFoundError: No module named 'src'`**

Make sure you're running from the project root:

```bash
cd sports_thesis
python main.py
```

**`KeyError: 'C'` or `KeyError: 'D'`**

Some questions have fewer than 4 options. The dataset.py handles this automatically — make sure you have the latest version of `src/dataset.py`.

**`CUDA out of memory`**

Reduce batch size in `main.py`:

```python
BATCH_SIZE = 2  # reduce from 4 to 2
```

**`FileNotFoundError: data/SportU_Video_mc.json`**

Download the annotation file from the SPORTU GitHub and place it in the `data/` folder.

**Model downloads taking long (CLIP, ResNet)**

This only happens once. Models are cached after the first download. Make sure `HF_HOME` points to a drive with enough space (at least 2GB free).

---

## 📚 References

- [SPORTU Dataset — ICLR 2025](https://github.com/chili-lab/SPORTU)
- [CLIP — OpenAI](https://github.com/openai/CLIP)
- [ResNet — Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [PyTorch](https://pytorch.org)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)

---

## 👤 Author

Tamim — BSc Computer Science and Engineering

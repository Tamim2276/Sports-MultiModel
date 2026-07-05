import torch
from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    
    total_loss    = 0.0
    total_correct = 0
    total_samples = 0
    
    for batch in tqdm(loader, desc="Training"):
        frames  = batch['frames'].to(device)
        ids     = batch['input_ids'].to(device)
        mask    = batch['attention_mask'].to(device)
        labels  = batch['label'].to(device)
        
        optimizer.zero_grad()
        scores = model(frames, ids, mask)
        loss   = criterion(scores, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )
        optimizer.step()
        
        preds          = scores.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        total_loss    += loss.item()
    
    return (
        total_loss / len(loader),
        total_correct / total_samples * 100
    )


def evaluate(model, loader, criterion, device):
    model.eval()
    
    total_loss    = 0.0
    total_correct = 0
    total_samples = 0
    all_preds     = []
    all_labels    = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            frames  = batch['frames'].to(device)
            ids     = batch['input_ids'].to(device)
            mask    = batch['attention_mask'].to(device)
            labels  = batch['label'].to(device)
            
            scores = model(frames, ids, mask)
            loss   = criterion(scores, labels)
            preds  = scores.argmax(dim=1)
            
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            total_loss    += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return (
        total_loss / len(loader),
        total_correct / total_samples * 100,
        all_preds,
        all_labels
    )
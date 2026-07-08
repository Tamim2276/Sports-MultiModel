import os
import json
import torch
import torch.nn as nn
import torch_directml
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from transformers import RobertaTokenizer
from torch.optim.lr_scheduler import OneCycleLR
from src.dataset import SPORTUDataset, TRAIN_TRANSFORM, EVAL_TRANSFORM
from src.model   import SportsMultimodalModel
from src.train   import train_one_epoch, evaluate

# Settings
DATA_PATH   = 'data/SportU_Video_mc.json'
VIDEO_DIR   = 'videos/Soccer'
OUTPUT_DIR  = 'outputs'
BATCH_SIZE  = 4
NUM_EPOCHS  = 20
FEATURE_DIM = 768
NUM_FRAMES  = 8


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Device
    device = torch_directml.device()
    print(f"Using Arc B580 GPU via DirectML!")

    # Load data
    with open(DATA_PATH) as f:
        data = json.load(f)

    # Processor — must match roberta-base used in TextEncoder
    processor = RobertaTokenizer.from_pretrained('roberta-base')

    # Eval dataset instance — no augmentation, used for val and test
    full_dataset_eval = SPORTUDataset(
        data       = data,
        frames_dir = 'data/frames',
        processor  = processor,
        num_frames = NUM_FRAMES,
        transform  = EVAL_TRANSFORM
    )

    # Split 70% train, 15% val, 15% test
    total         = len(full_dataset_eval)
    n_train       = int(0.70 * total)
    n_val         = int(0.15 * total)
    n_test        = total - n_train - n_val

    indices       = torch.randperm(total, generator=torch.Generator().manual_seed(42)).tolist()
    train_indices = indices[:n_train]
    val_indices   = indices[n_train:n_train + n_val]
    test_indices  = indices[n_train + n_val:]

    # Train dataset instance — with augmentation
    full_dataset_train = SPORTUDataset(
        data       = data,
        frames_dir = 'data/frames',
        processor  = processor,
        num_frames = NUM_FRAMES,
        transform  = TRAIN_TRANSFORM
    )

    train_set = Subset(full_dataset_train, train_indices)
    val_set   = Subset(full_dataset_eval,  val_indices)
    test_set  = Subset(full_dataset_eval,  test_indices)

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE,
        shuffle=True,  num_workers=0
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_set, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0
    )

    print(f"Train: {n_train}  Val: {n_val}  Test: {n_test}")

    # Model
    model     = SportsMultimodalModel(FEATURE_DIM).to(device)
    # Label smoothing in the loss
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) 

    # Differential learning rates:
    # Pretrained backbone layers small LR to preserve learned weights
    # Projection, fusion, classifier higher LR, training from scratch
    # weight_decay added to all groups to penalize large weights (reduces overfitting)
    optimizer = optim.AdamW([
        {'params': model.video_encoder.vit.encoder.layer[-6:].parameters(),    'lr': 1e-5, 'weight_decay': 1e-4},
        {'params': model.text_encoder.roberta.encoder.layer[-6:].parameters(), 'lr': 1e-5, 'weight_decay': 1e-4},
        {'params': model.video_encoder.projection.parameters(),                 'lr': 1e-4, 'weight_decay': 1e-4},
        {'params': model.text_encoder.projection.parameters(),                  'lr': 1e-4, 'weight_decay': 1e-4},
        {'params': model.fusion.parameters(),                                    'lr': 1e-4, 'weight_decay': 1e-4},
        {'params': model.classifier.parameters(),                                'lr': 1e-4, 'weight_decay': 1e-4},
    ])

    #scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr          = [1e-5, 1e-5, 1e-4, 1e-4, 1e-4, 1e-4],
        steps_per_epoch = len(train_loader),
        epochs          = NUM_EPOCHS,
        pct_start       = 0.1
    )

    trainable = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )
    print(f"Trainable parameters: {trainable:,}")

    # Training
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    best_val_acc             = 0.0
    patience                 = 15    # stop if no improvement for 5 epochs
    patience_count           = 0

    print("\n" + "=" * 40)
    print("Starting Training")
    print("=" * 40)

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scheduler
        )
        vl_loss, vl_acc, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        train_accs.append(tr_acc)
        val_accs.append(vl_acc)

        print(f"  Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.2f}%")
        print(f"  Val   Loss: {vl_loss:.4f}  Acc: {vl_acc:.2f}%")

        # Save best model and track early stopping patience
        if vl_acc > best_val_acc:
            best_val_acc   = vl_acc
            patience_count = 0
            torch.save(
                model.state_dict(),
                os.path.join(OUTPUT_DIR, 'best_model.pth')
            )
            print(f"  ✅ Best model saved!")
        else:
            patience_count += 1
            print(f"  No improvement. Patience: {patience_count}/{patience}")
            if patience_count >= patience:
                print(f"\n⛔ Early stopping triggered at epoch {epoch + 1}")
                break

    # Test — load best checkpoint
    print("\n" + "=" * 40)
    model.load_state_dict(
        torch.load(
            os.path.join(OUTPUT_DIR, 'best_model.pth'),
            map_location='cpu',
            weights_only=False
        )
    )
    ts_loss, ts_acc, _, _ = evaluate(
        model, test_loader, criterion, device
    )
    print(f"Test Accuracy: {ts_acc:.2f}%")

    # Plot
    epochs_ran = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs_ran, train_losses, 'b-o', label='Train')
    ax1.plot(epochs_ran, val_losses,   'r-o', label='Val')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs_ran, train_accs, 'b-o', label='Train')
    ax2.plot(epochs_ran, val_accs,   'r-o', label='Val')
    ax2.set_title('Accuracy (%)')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=150)
    plt.show()

    # Save results
    with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump({
            'best_val_accuracy': best_val_acc,
            'test_accuracy'    : ts_acc,
            'total_examples'   : total,
            'epochs_trained'   : len(train_losses)
        }, f, indent=2)

    print("\n✅ Complete!")
    print(f"Best Val Acc:  {best_val_acc:.2f}%")
    print(f"Test Accuracy: {ts_acc:.2f}%")
    print("Results saved to outputs/")


if __name__ == "__main__":
    main()
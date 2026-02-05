"""
Test and evaluate SatFormerNet model on validation set.
Calculates accuracy, mIoU, per-class metrics, and confusion matrix.
"""
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json

from model.satformernet import SatFormerNet
from train.dataset import SatelliteDataset
from train.augmentations import get_val_transforms


def calculate_metrics(pred, target, num_classes=5):
    """Calculate comprehensive segmentation metrics."""
    pred = pred.flatten()
    target = target.flatten()
    
    # Per-class IoU
    ious = []
    accs = []
    
    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        
        intersection = (pred_c & target_c).sum()
        union = (pred_c | target_c).sum()
        
        if union > 0:
            iou = intersection / union
            ious.append(iou.item())
        
        # Per-class accuracy
        if target_c.sum() > 0:
            acc = intersection / target_c.sum()
            accs.append(acc.item())
    
    # Overall pixel accuracy
    correct = (pred == target).sum()
    total = len(pred)
    pixel_acc = correct / total
    
    # Mean IoU
    miou = np.mean(ious) if ious else 0
    
    # Mean class accuracy
    mean_acc = np.mean(accs) if accs else 0
    
    return {
        'pixel_accuracy': pixel_acc.item(),
        'mean_iou': miou,
        'mean_class_accuracy': mean_acc,
        'per_class_iou': ious,
        'per_class_accuracy': accs
    }


def build_confusion_matrix(pred, target, num_classes=5):
    """Build confusion matrix."""
    pred = pred.flatten()
    target = target.flatten()
    
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(target, pred):
        if t < num_classes and p < num_classes:
            cm[t, p] += 1
    
    return cm


@torch.no_grad()
def evaluate_model(
    model, 
    dataloader, 
    device, 
    num_classes=5,
    class_names=None
):
    """Run full evaluation on dataset."""
    model.eval()
    
    all_preds = []
    all_targets = []
    
    total_correct = 0
    total_pixels = 0
    
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    
    print("\nEvaluating model on validation set...")
    
    for images, masks in tqdm(dataloader, desc="Evaluating"):
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        outputs = model(images)
        logits = outputs['out'] if isinstance(outputs, dict) else outputs
        
        # Get predictions
        preds = logits.argmax(dim=1)
        
        # Clamp masks to valid range
        masks = masks.clamp(0, num_classes - 1)
        
        # Accumulate
        total_correct += (preds == masks).sum().item()
        total_pixels += masks.numel()
        
        # Build confusion matrix
        for pred, target in zip(preds.cpu(), masks.cpu()):
            cm = build_confusion_matrix(pred, target, num_classes)
            confusion_matrix += cm
        
        all_preds.append(preds.cpu())
        all_targets.append(masks.cpu())
    
    # Combine all predictions
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate metrics
    metrics = calculate_metrics(all_preds, all_targets, num_classes)
    
    # Overall pixel accuracy
    overall_accuracy = total_correct / total_pixels * 100
    
    # Class names
    if class_names is None:
        class_names = ['Background', 'Roads', 'Water', 'Trees', 'Buildings']
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print(f"\n{'Metric':<30} {'Value':>15}")
    print("-"*45)
    print(f"{'Overall Pixel Accuracy':<30} {overall_accuracy:>14.2f}%")
    print(f"{'Mean Class Accuracy':<30} {metrics['mean_class_accuracy']*100:>14.2f}%")
    print(f"{'Mean IoU (mIoU)':<30} {metrics['mean_iou']*100:>14.2f}%")
    
    print("\n" + "-"*60)
    print("PER-CLASS METRICS")
    print("-"*60)
    print(f"{'Class':<15} {'IoU':>12} {'Accuracy':>15}")
    print("-"*45)
    
    for i, name in enumerate(class_names):
        if i < len(metrics['per_class_iou']):
            iou = metrics['per_class_iou'][i] * 100
            acc = metrics['per_class_accuracy'][i] * 100 if i < len(metrics['per_class_accuracy']) else 0
            print(f"{name:<15} {iou:>11.2f}% {acc:>14.2f}%")
    
    print("\n" + "-"*60)
    print("CONFUSION MATRIX")
    print("-"*60)
    
    # Print confusion matrix header
    print(f"{'Predicted →':>12}", end='')
    for name in class_names:
        print(f"{name[:8]:>10}", end='')
    print()
    print("Actual ↓")
    
    for i, name in enumerate(class_names):
        print(f"{name[:10]:<12}", end='')
        for j in range(num_classes):
            val = confusion_matrix[i, j].item()
            if val > 1000000:
                print(f"{val/1e6:>9.1f}M", end='')
            elif val > 1000:
                print(f"{val/1e3:>9.1f}K", end='')
            else:
                print(f"{val:>10}", end='')
        print()
    
    # Save results
    results = {
        'overall_accuracy': overall_accuracy,
        'mean_class_accuracy': metrics['mean_class_accuracy'] * 100,
        'mean_iou': metrics['mean_iou'] * 100,
        'per_class_iou': [x * 100 for x in metrics['per_class_iou']],
        'per_class_accuracy': [x * 100 for x in metrics['per_class_accuracy']],
        'class_names': class_names,
        'confusion_matrix': confusion_matrix.tolist()
    }
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate SatFormerNet model')
    parser.add_argument('--checkpoint', type=str, default='./outputs/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, 
                       default=r'C:\Users\sagni\.cache\kagglehub\datasets\humansintheloop\semantic-segmentation-of-aerial-imagery\versions\1\Semantic segmentation dataset',
                       help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--num_classes', type=int, default=5)
    
    args = parser.parse_args()
    
    print("="*60)
    print("SatFormerNet Model Evaluation")
    print("="*60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    model = SatFormerNet(
        num_classes=args.num_classes,
        in_channels=3,
        backbone_channels=[96, 192, 384, 768],
        decoder_channels=[256, 128, 64, 32],
        use_pretrained=False,
        use_adapters=True,
        use_msda=False,
        use_baam=True,
        deep_supervision=True,
        dropout=0.1
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dataset
    print(f"\nLoading dataset from: {args.data_path}")
    
    dataset = SatelliteDataset(
        root=args.data_path,
        split='val',
        transform=get_val_transforms(size=(args.image_size, args.image_size)),
        image_size=(args.image_size, args.image_size)
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"Validation samples: {len(dataset)}")
    
    # Evaluate
    results = evaluate_model(
        model, 
        dataloader, 
        device, 
        num_classes=args.num_classes,
        class_names=['Background', 'Roads', 'Water', 'Trees', 'Buildings']
    )
    
    # Save results
    output_path = Path(args.checkpoint).parent / 'evaluation_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    print("\n" + "="*60)
    print("SUMMARY")  
    print("="*60)
    print(f"  Overall Accuracy: {results['overall_accuracy']:.2f}%")
    print(f"  Mean IoU: {results['mean_iou']:.2f}%")
    print(f"  Mean Class Accuracy: {results['mean_class_accuracy']:.2f}%")
    print("="*60)


if __name__ == "__main__":
    main()

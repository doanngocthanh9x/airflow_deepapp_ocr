"""
Document Classifier Training - Tasks
Prepare data, train, evaluate, export
"""

import os
import sys
import json
import shutil
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_dataset(**context) -> Dict:
    """
    Task 1: Prepare dataset - split into train/val/test
    Creates dataset structure compatible with PaddleClas
    """
    from dag_train_classifier.config import (
        DATA_DIR, OUTPUT_DIR, TRAIN_CONFIG, CLASSES, MODEL_CONFIG
    )
    
    logger.info("=" * 60)
    logger.info("TASK: Prepare Dataset")
    logger.info("=" * 60)
    
    # Create output directories
    dataset_dir = OUTPUT_DIR / 'dataset'
    for split in ['train', 'val', 'test']:
        (dataset_dir / split).mkdir(parents=True, exist_ok=True)
    
    # Collect all images per class
    class_images = {}
    for cls in CLASSES:
        cls_dir = DATA_DIR / cls
        if cls_dir.exists():
            images = list(cls_dir.glob('*.jpg')) + \
                     list(cls_dir.glob('*.jpeg')) + \
                     list(cls_dir.glob('*.png'))
            class_images[cls] = images
            logger.info(f"  {cls}: {len(images)} images")
    
    # Split data
    train_ratio = TRAIN_CONFIG['train_ratio']
    val_ratio = TRAIN_CONFIG['val_ratio']
    
    train_list = []
    val_list = []
    test_list = []
    
    for cls_idx, cls in enumerate(CLASSES):
        images = class_images.get(cls, [])
        random.shuffle(images)
        
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        # Copy images and create lists
        for img in train_images:
            dst = dataset_dir / 'train' / cls / img.name
            dst.parent.mkdir(exist_ok=True)
            if not dst.exists():
                shutil.copy2(img, dst)
            train_list.append(f"{dst.relative_to(dataset_dir)} {cls_idx}")
        
        for img in val_images:
            dst = dataset_dir / 'val' / cls / img.name
            dst.parent.mkdir(exist_ok=True)
            if not dst.exists():
                shutil.copy2(img, dst)
            val_list.append(f"{dst.relative_to(dataset_dir)} {cls_idx}")
        
        for img in test_images:
            dst = dataset_dir / 'test' / cls / img.name
            dst.parent.mkdir(exist_ok=True)
            if not dst.exists():
                shutil.copy2(img, dst)
            test_list.append(f"{dst.relative_to(dataset_dir)} {cls_idx}")
        
        logger.info(f"  {cls}: train={len(train_images)}, val={len(val_images)}, test={len(test_images)}")
    
    # Save train/val/test lists (PaddleClas format)
    with open(dataset_dir / 'train_list.txt', 'w') as f:
        f.write('\n'.join(train_list))
    
    with open(dataset_dir / 'val_list.txt', 'w') as f:
        f.write('\n'.join(val_list))
    
    with open(dataset_dir / 'test_list.txt', 'w') as f:
        f.write('\n'.join(test_list))
    
    # Save label map
    label_map = {i: cls for i, cls in enumerate(CLASSES)}
    with open(dataset_dir / 'label_map.json', 'w', encoding='utf-8') as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)
    
    result = {
        'dataset_dir': str(dataset_dir),
        'num_classes': len(CLASSES),
        'classes': CLASSES,
        'train_count': len(train_list),
        'val_count': len(val_list),
        'test_count': len(test_list),
    }
    
    logger.info(f"\nDataset prepared:")
    logger.info(f"  Train: {len(train_list)} images")
    logger.info(f"  Val: {len(val_list)} images")
    logger.info(f"  Test: {len(test_list)} images")
    logger.info(f"  Output: {dataset_dir}")
    
    return result


def train_model(**context) -> Dict:
    """
    Task 2: Train PPLCNetV2 model using PaddleClas
    """
    import paddle
    import paddle.nn as nn
    from paddle.io import DataLoader, Dataset
    from paddle.vision import transforms
    from PIL import Image
    import numpy as np
    
    from dag_train_classifier.config import (
        OUTPUT_DIR, MODEL_CONFIG, TRAIN_CONFIG, AUGMENT_CONFIG, LOGS_DIR
    )
    
    logger.info("=" * 60)
    logger.info("TASK: Train Model")
    logger.info("=" * 60)
    
    # Get dataset info from previous task
    ti = context.get('ti')
    if ti:
        dataset_info = ti.xcom_pull(task_ids='prepare_dataset')
    else:
        # Running standalone
        dataset_info = {
            'dataset_dir': str(OUTPUT_DIR / 'dataset'),
            'num_classes': MODEL_CONFIG['num_classes'],
            'classes': [],
        }
    
    dataset_dir = Path(dataset_info['dataset_dir'])
    num_classes = dataset_info['num_classes']
    
    logger.info(f"Dataset: {dataset_dir}")
    logger.info(f"Num classes: {num_classes}")
    logger.info(f"Device: {paddle.device.get_device()}")
    
    # ========================================================================
    # Custom Dataset
    # ========================================================================
    class DocDataset(Dataset):
        def __init__(self, list_file, root_dir, transform=None):
            self.root_dir = Path(root_dir)
            self.transform = transform
            self.samples = []
            
            with open(list_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        img_path = parts[0]
                        label = int(parts[-1])
                        self.samples.append((img_path, label))
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            img_path, label = self.samples[idx]
            img = Image.open(self.root_dir / img_path).convert('RGB')
            
            if self.transform:
                img = self.transform(img)
            
            return img, label
    
    # ========================================================================
    # Transforms
    # ========================================================================
    img_size = MODEL_CONFIG['image_size']
    
    train_transform = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomRotation(AUGMENT_CONFIG['random_rotation']),
        transforms.ColorJitter(
            brightness=AUGMENT_CONFIG['brightness'],
            contrast=AUGMENT_CONFIG['contrast'],
            saturation=AUGMENT_CONFIG['saturation']
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # ========================================================================
    # DataLoaders
    # ========================================================================
    train_dataset = DocDataset(
        dataset_dir / 'train_list.txt', 
        dataset_dir, 
        train_transform
    )
    val_dataset = DocDataset(
        dataset_dir / 'val_list.txt', 
        dataset_dir, 
        val_transform
    )
    
    batch_size = TRAIN_CONFIG['batch_size']
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # ========================================================================
    # Model - MobileNetV3 with LOCAL pretrained weights
    # ========================================================================
    from paddle.vision.models import mobilenet_v3_small
    
    # Load model WITHOUT pretrained (no download)
    backbone = mobilenet_v3_small(pretrained=False)
    
    # Get in_features BEFORE loading pretrained (original shape)
    # In Paddle, Linear weight shape is (in_features, out_features)
    in_features = backbone.classifier[3].weight.shape[0]  # = 1024
    logger.info(f"Classifier in_features: {in_features}")
    
    # Replace classifier head for our number of classes FIRST
    backbone.classifier[3] = nn.Linear(in_features, num_classes)
    logger.info(f"Replaced classifier head: Linear({in_features}, {num_classes})")
    
    # Load local pretrained weights (skip classifier layer)
    pretrained_path = MODEL_CONFIG.get('pretrained_path')
    if pretrained_path and Path(pretrained_path).exists():
        logger.info(f"Loading pretrained weights from: {pretrained_path}")
        state_dict = paddle.load(pretrained_path)
        
        # Remove classifier.3 weights (different shape due to num_classes)
        keys_to_remove = [k for k in state_dict.keys() if 'classifier.3' in k]
        for k in keys_to_remove:
            logger.info(f"  Skipping pretrained key: {k}")
            del state_dict[k]
        
        # Load remaining weights
        backbone.set_state_dict(state_dict)
        logger.info("Pretrained backbone weights loaded successfully!")
    else:
        logger.warning(f"Pretrained weights not found: {pretrained_path}")
        logger.warning("Training from scratch (may take longer)")
    
    model = backbone
    logger.info(f"Model: MobileNetV3-Small with {num_classes} classes")
    
    # ====================================================================
    # Training loop
    # ====================================================================
    epochs = TRAIN_CONFIG['epochs']
    lr = TRAIN_CONFIG['learning_rate']
    
    # Scheduler
    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
        learning_rate=lr,
        T_max=epochs,
        eta_min=1e-6
    )
    
    # Warmup
    warmup_epochs = TRAIN_CONFIG['warmup_epochs']
    scheduler = paddle.optimizer.lr.LinearWarmup(
        learning_rate=scheduler,
        warmup_steps=warmup_epochs * len(train_loader),
        start_lr=lr / 10,
        end_lr=lr
    )
    
    optimizer = paddle.optimizer.Adam(
        learning_rate=scheduler,
        parameters=model.parameters(),
        weight_decay=TRAIN_CONFIG['weight_decay']
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Training
    best_acc = 0.0
    best_epoch = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    save_dir = OUTPUT_DIR / 'checkpoints'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            scheduler.step()
            
            train_loss += loss.item()
            preds = paddle.argmax(outputs, axis=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.shape[0]
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with paddle.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                preds = paddle.argmax(outputs, axis=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.shape[0]
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        logger.info(f"Epoch {epoch+1}/{epochs}:")
        logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            paddle.save(model.state_dict(), str(save_dir / 'best_model.pdparams'))
            logger.info(f"  -> New best model saved! Acc: {best_acc:.4f}")
        
        # Early stopping
        if epoch - best_epoch >= TRAIN_CONFIG['early_stop_patience']:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Save final model
    paddle.save(model.state_dict(), str(save_dir / 'final_model.pdparams'))
    
    # Save training history
    with open(save_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    result = {
        'model_path': str(save_dir / 'best_model.pdparams'),
        'best_epoch': best_epoch,
        'best_accuracy': best_acc,
        'history': history,
    }
    
    return result


def evaluate_model(**context) -> Dict:
    """
    Task 3: Evaluate model on test set
    """
    import paddle
    from paddle.io import DataLoader, Dataset
    from paddle.vision import transforms
    from PIL import Image
    import numpy as np
    from sklearn.metrics import classification_report, confusion_matrix
    
    from dag_train_classifier.config import OUTPUT_DIR, MODEL_CONFIG, CLASSES
    
    logger.info("=" * 60)
    logger.info("TASK: Evaluate Model")
    logger.info("=" * 60)
    
    # Get paths
    ti = context.get('ti')
    if ti:
        train_result = ti.xcom_pull(task_ids='train_model')
        dataset_info = ti.xcom_pull(task_ids='prepare_dataset')
    else:
        train_result = {'model_path': str(OUTPUT_DIR / 'checkpoints' / 'best_model.pdparams')}
        dataset_info = {'dataset_dir': str(OUTPUT_DIR / 'dataset')}
    
    dataset_dir = Path(dataset_info['dataset_dir'])
    model_path = train_result['model_path']
    
    # Load model
    from paddle.vision.models import mobilenet_v3_small
    
    model = mobilenet_v3_small(pretrained=False)
    # In Paddle, Linear weight shape is (in_features, out_features)
    in_features = model.classifier[3].weight.shape[0]  # = 1024
    model.classifier[3] = paddle.nn.Linear(in_features, len(CLASSES))
    
    if Path(model_path).exists():
        model.set_state_dict(paddle.load(model_path))
        logger.info(f"Loaded model from: {model_path}")
    
    model.eval()
    
    # Prepare test data
    img_size = MODEL_CONFIG['image_size']
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Simple dataset for evaluation
    class TestDataset(Dataset):
        def __init__(self, list_file, root_dir, transform):
            self.root_dir = Path(root_dir)
            self.transform = transform
            self.samples = []
            with open(list_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        self.samples.append((parts[0], int(parts[-1])))
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            img_path, label = self.samples[idx]
            img = Image.open(self.root_dir / img_path).convert('RGB')
            img = self.transform(img)
            return img, label
    
    test_dataset = TestDataset(dataset_dir / 'test_list.txt', dataset_dir, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Evaluate
    all_preds = []
    all_labels = []
    
    with paddle.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            preds = paddle.argmax(outputs, axis=1)
            all_preds.extend(preds.numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
    
    # Metrics
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_preds)
    
    report = classification_report(
        all_labels, all_preds, 
        target_names=CLASSES, 
        output_dict=True
    )
    
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # Save results
    eval_dir = OUTPUT_DIR / 'evaluation'
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    with open(eval_dir / 'classification_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    np.save(eval_dir / 'confusion_matrix.npy', conf_matrix)
    
    logger.info(f"\nTest Accuracy: {accuracy:.4f}")
    logger.info(f"\nClassification Report:")
    logger.info(classification_report(all_labels, all_preds, target_names=CLASSES))
    
    return {
        'accuracy': accuracy,
        'report': report,
        'eval_dir': str(eval_dir),
    }


def export_inference_model(**context) -> Dict:
    """
    Task 4: Export model for inference
    """
    import paddle
    from paddle.vision.models import mobilenet_v3_small
    
    from dag_train_classifier.config import OUTPUT_DIR, MODEL_CONFIG, CLASSES
    
    logger.info("=" * 60)
    logger.info("TASK: Export Inference Model")
    logger.info("=" * 60)
    
    # Load best model
    model_path = OUTPUT_DIR / 'checkpoints' / 'best_model.pdparams'
    
    model = mobilenet_v3_small(pretrained=False)
    # In Paddle, Linear weight shape is (in_features, out_features)
    in_features = model.classifier[3].weight.shape[0]  # = 1024
    model.classifier[3] = paddle.nn.Linear(in_features, len(CLASSES))
    
    if model_path.exists():
        model.set_state_dict(paddle.load(str(model_path)))
    
    model.eval()
    
    # Export to static graph
    export_dir = OUTPUT_DIR / 'inference'
    export_dir.mkdir(parents=True, exist_ok=True)
    
    img_size = MODEL_CONFIG['image_size']
    input_spec = paddle.static.InputSpec(
        shape=[None, 3, img_size, img_size],
        dtype='float32',
        name='image'
    )
    
    static_model = paddle.jit.to_static(model, input_spec=[input_spec])
    paddle.jit.save(static_model, str(export_dir / 'model'))
    
    # Save label map
    label_map = {i: cls for i, cls in enumerate(CLASSES)}
    with open(export_dir / 'label_map.json', 'w', encoding='utf-8') as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)
    
    # Save config
    config = {
        'model_name': MODEL_CONFIG['model_name'],
        'image_size': img_size,
        'num_classes': len(CLASSES),
        'classes': CLASSES,
        'normalize': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }
    }
    with open(export_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Model exported to: {export_dir}")
    logger.info(f"Files: model.pdmodel, model.pdiparams, label_map.json, config.json")
    
    return {
        'export_dir': str(export_dir),
        'model_file': str(export_dir / 'model.pdmodel'),
        'params_file': str(export_dir / 'model.pdiparams'),
    }

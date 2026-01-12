import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import json

from src.data.dataset import VOCDetectionDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.detector import GridDetector
from src.training.loss import DetectionLoss


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        os.makedirs(config['log_dir'], exist_ok=True)
        
        # Dataset & DataLoader
        self.train_dataset = VOCDetectionDataset(
            img_dir=config['train_img_dir'],
            ann_dir=config['train_ann_dir'],
            transform=get_train_transforms(config['img_size']),
            max_objects=config['max_objects']
        )
        
        self.val_dataset = VOCDetectionDataset(
            img_dir=config['val_img_dir'],
            ann_dir=config['val_ann_dir'],
            transform=get_val_transforms(config['img_size']),
            max_objects=config['max_objects']
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=(self.device.type == "cuda"),
            persistent_workers=True if config['num_workers'] > 0 else False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=(self.device.type == "cuda"),
            persistent_workers=True if config['num_workers'] > 0 else False
        )
        
        # Model
        self.model = GridDetector(
            num_classes=config['num_classes'],
            grid_size=config['grid_size'],
            anchors_per_cell=config['anchors_per_cell']
        ).to(self.device)
        
        # Loss
        self.criterion = DetectionLoss(
            num_classes=config['num_classes'],
            grid_size=config['grid_size'],
            anchors_per_cell=config['anchors_per_cell']
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config['num_epochs'],
            eta_min=1e-6
        )
        
        # Mixed precision
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        self.warmup_epochs = config.get('warmup_epochs', 5)
        
        # Training state
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        # ✅ ADDED: Early stopping
        self.patience = config.get('patience', 15)  # Stop if no improvement for 15 epochs
        self.epochs_without_improvement = 0
        self.early_stopped = False
        
        if config.get('resume'):
            self.load_checkpoint(config['resume'])
        
        print(f"\n✅ Training setup complete (3 CLASSES)!")
        print(f"   Train samples: {len(self.train_dataset)}")
        print(f"   Val samples: {len(self.val_dataset)}")
        print(f"   Batch size: {config['batch_size']}")
        print(f"   Mixed precision: {'ON' if self.use_amp else 'OFF'}")
    
    def train_epoch(self):
        self.model.train()
        
        total_loss = 0
        loss_components = {'bbox': 0, 'obj': 0, 'noobj': 0, 'cls': 0}
        
        pbar = tqdm(self.train_loader, desc="Training")
        
        for images, target_boxes, target_labels, num_objs in pbar:
            images = images.to(self.device)
            target_boxes = target_boxes.to(self.device)
            target_labels = target_labels.to(self.device)
            num_objs = num_objs.to(self.device)
            
            if self.use_amp:
                with autocast():
                    predictions = self.model(images)
                    loss, loss_dict = self.criterion(predictions, target_boxes, target_labels, num_objs)
                
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(images)
                loss, loss_dict = self.criterion(predictions, target_boxes, target_labels, num_objs)
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.optimizer.step()
            
            total_loss += loss.item()
            for key in loss_components:
                loss_components[key] += loss_dict[f'loss_{key}']
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'bbox': f"{loss_dict['loss_bbox']:.4f}"
            })
        
        avg_loss = total_loss / len(self.train_loader)
        for key in loss_components:
            loss_components[key] /= len(self.train_loader)
        
        return avg_loss, loss_components
    
    def validate(self):
        self.model.eval()
        
        total_loss = 0
        loss_components = {'bbox': 0, 'obj': 0, 'noobj': 0, 'cls': 0}
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            
            for images, target_boxes, target_labels, num_objs in pbar:
                images = images.to(self.device)
                target_boxes = target_boxes.to(self.device)
                target_labels = target_labels.to(self.device)
                num_objs = num_objs.to(self.device)
                
                if self.use_amp:
                    with autocast():
                        predictions = self.model(images)
                        loss, loss_dict = self.criterion(predictions, target_boxes, target_labels, num_objs)
                else:
                    predictions = self.model(images)
                    loss, loss_dict = self.criterion(predictions, target_boxes, target_labels, num_objs)
                
                total_loss += loss.item()
                for key in loss_components:
                    loss_components[key] += loss_dict[f'loss_{key}']
                
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(self.val_loader)
        for key in loss_components:
            loss_components[key] /= len(self.val_loader)
        
        return avg_loss, loss_components
    
    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': self.config
        }
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        path = os.path.join(self.config['checkpoint_dir'], 'latest.pth')
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = os.path.join(self.config['checkpoint_dir'], 'best.pth')
            torch.save(checkpoint, best_path)
            print(f"💾 Best model saved at epoch {epoch+1}")
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"✅ Resumed from epoch {self.start_epoch}")
    
    def save_history(self):
        history_path = os.path.join(self.config['log_dir'], 'history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def train(self):
        print(f"\n{'='*70}")
        print(f"TRAINING - 3 CLASSES (FAST VERSION)")
        print(f"{'='*70}")
        print(f"  Device: {self.device}")
        print(f"  Epochs: {self.config['num_epochs']}")
        print(f"  Batch size: {self.config['batch_size']}")
        print(f"  Learning rate: {self.config['learning_rate']}")
        print(f"  Mixed precision: {'ON' if self.use_amp else 'OFF'}")
        print(f"  Early stopping patience: {self.patience} epochs")  # ✅ Added
        print(f"{'='*70}\n")
        
        for epoch in range(self.start_epoch, self.config['num_epochs']):
            print(f"\n{'='*70}")
            print(f"Epoch [{epoch+1}/{self.config['num_epochs']}]")
            print(f"{'='*70}")
            
            train_loss, train_components = self.train_epoch()
            val_loss, val_components = self.validate()
            
            if epoch < self.warmup_epochs:
                lr_scale = (epoch + 1) / self.warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.config['learning_rate'] * lr_scale
            else:
                self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)
            
            print(f"\n📊 Results:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0  # ✅ Reset counter
                print(f"  ⭐ New best validation loss!")
            else:
                self.epochs_without_improvement += 1  # ✅ Increment counter
            
            self.save_checkpoint(epoch, is_best)
            self.save_history()
            
            # ✅ EARLY STOPPING CHECK
            if self.epochs_without_improvement >= self.patience:
                print(f"\n{'='*70}")
                print(f"⏹️  EARLY STOPPING TRIGGERED!")
                print(f"{'='*70}")
                print(f"No improvement for {self.patience} epochs")
                print(f"Best validation loss: {self.best_val_loss:.4f}")
                print(f"Stopping at epoch {epoch+1}/{self.config['num_epochs']}")
                print(f"{'='*70}")
                self.early_stopped = True
                break
        
        print(f"\n{'='*70}")
        if self.early_stopped:
            print(f"⏹️  TRAINING STOPPED EARLY!")
            print(f"Reason: No improvement for {self.patience} epochs")
        else:
            print(f"✅ TRAINING COMPLETED!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'='*70}\n")


def main():
    # ✅ 3 CLASSES CONFIGURATION
    config = {
        'train_img_dir': 'dataset/train/images',
        'train_ann_dir': 'dataset/train/annotations',
        'val_img_dir': 'dataset/val/images',
        'val_ann_dir': 'dataset/val/annotations',
        
        'num_classes': 3,       # ✅ person, car, chair
        'grid_size': 7,
        'anchors_per_cell': 2,
        'img_size': 224,
        'max_objects': 30,
        
        'num_epochs': 120,      # ✅ Reduced (faster training)
        'batch_size': 6,
        'num_workers': 0,
        'warmup_epochs': 5,
        'use_amp': True,
        'patience': 15,         # ✅ Early stopping: stop if no improvement for 15 epochs
        
        'learning_rate': 1e-3,
        'weight_decay': 5e-4,
        
        'checkpoint_dir': 'outputs/weights',
        'log_dir': 'outputs/logs',
        
        'resume': None
    }
    
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
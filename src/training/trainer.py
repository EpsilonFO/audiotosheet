"""Module pour l'entraînement des modèles."""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from tqdm import tqdm

from src.training.dataset import AudioToMIDIDataset
from src.training.metrics import calculate_metrics
import config


class Trainer:
    """Classe pour gérer l'entraînement des modèles."""
    
    def __init__(self, model, data_dir, checkpoint_dir, batch_size=16, 
                 learning_rate=0.001, num_epochs=50, device="cuda"):
        """
        Initialiser le gestionnaire d'entraînement.
        
        Args:
            model: Modèle à entraîner
            data_dir: Répertoire contenant les données prétraitées
            checkpoint_dir: Répertoire pour sauvegarder les points de contrôle
            batch_size: Taille du batch
            learning_rate: Taux d'apprentissage
            num_epochs: Nombre d'époques d'entraînement
            device: Appareil d'entraînement ("cpu" ou "cuda")
        """
        self.model = model
        self.data_dir = Path(data_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = device
        
        # Créer le répertoire de points de contrôle s'il n'existe pas
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Définir la fonction de perte (Binary Cross Entropy pour multi-étiquettes)
        self.criterion = nn.BCELoss()
        
        # Définir l'optimiseur
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Définir un scheduler pour réduire le taux d'apprentissage
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
    def create_dataloaders(self):
        """
        Créer les dataloaders pour l'entraînement et la validation.
        
        Returns:
            tuple: (train_loader, val_loader)
        """
        # Créer le dataset complet
        full_dataset = AudioToMIDIDataset(
            data_dir=self.data_dir, 
            num_frames=None,  # Utiliser toutes les trames disponibles
            augmentation=False  # Pas d'augmentation pour l'instant
        )
        
        # Diviser en ensembles d'entraînement et de validation (80% / 20%)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # Pour reproductibilité
        )
        
        # Activer l'augmentation sur l'ensemble d'entraînement uniquement
        train_dataset.dataset.augmentation = True
        
        # Créer les dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True if self.device == "cuda" else False
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if self.device == "cuda" else False
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader):
        """
        Entraîner le modèle pendant une époque.
        
        Args:
            train_loader: DataLoader pour l'ensemble d'entraînement
            
        Returns:
            dict: Métriques d'entraînement
        """
        self.model.train()
        
        epoch_loss = 0.0
        all_metrics = []
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (features, targets) in enumerate(pbar):
            # Transférer les données sur le device
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            # Réinitialiser les gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(features)
            
            # Calculer la perte
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Mettre à jour les poids
            self.optimizer.step()
            
            # Accumuler la perte
            epoch_loss += loss.item()
            
            # Calculer les métriques de performance
            metrics = calculate_metrics(outputs.detach().cpu(), targets.cpu())
            all_metrics.append(metrics)
            
            # Mettre à jour la barre de progression
            pbar.set_postfix(loss=loss.item(), f1=metrics["f1"])
        
        # Calculer les moyennes
        avg_loss = epoch_loss / len(train_loader)
        avg_metrics = {key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0].keys()}
        avg_metrics["loss"] = avg_loss
        
        return avg_metrics
    
    def validate(self, val_loader):
        """
        Valider le modèle.
        
        Args:
            val_loader: DataLoader pour l'ensemble de validation
            
        Returns:
            dict: Métriques de validation
        """
        self.model.eval()
        
        val_loss = 0.0
        all_metrics = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for batch_idx, (features, targets) in enumerate(pbar):
                # Transférer les données sur le device
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(features)
                
                # Calculer la perte
                loss = self.criterion(outputs, targets)
                
                # Accumuler la perte
                val_loss += loss.item()
                
                # Calculer les métriques de performance
                metrics = calculate_metrics(outputs.cpu(), targets.cpu())
                all_metrics.append(metrics)
                
                # Mettre à jour la barre de progression
                pbar.set_postfix(loss=loss.item(), f1=metrics["f1"])
        
        # Calculer les moyennes
        avg_loss = val_loss / len(val_loader)
        avg_metrics = {key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0].keys()}
        avg_metrics["loss"] = avg_loss
        
        return avg_metrics
    
    def train(self):
        """
        Entraîner le modèle pendant plusieurs époques.
        
        Returns:
            dict: Historique d'entraînement
        """
        # Créer les dataloaders
        train_loader, val_loader = self.create_dataloaders()
        
        # Initialiser l'historique d'entraînement
        history = {
            "train": [],
            "val": []
        }
        
        # Initialiser le suivi de la meilleure performance
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Entraîner pendant le nombre d'époques spécifié
        for epoch in range(self.num_epochs):
            print(f"Époque {epoch+1}/{self.num_epochs}")
            
            # Entraîner une époque
            train_metrics = self.train_epoch(train_loader)
            
            # Valider
            val_metrics = self.validate(val_loader)
            
            # Mettre à jour le learning rate
            self.scheduler.step(val_metrics["loss"])
            
            # Ajouter les métriques à l'historique
            history["train"].append(train_metrics)
            history["val"].append(val_metrics)
            
            # Afficher les métriques
            print(f"Train Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1']:.4f}")
            
            # Vérifier s'il s'agit de la meilleure performance
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
                
                # Sauvegarder le meilleur modèle
                checkpoint_path = self.checkpoint_dir / "best_model.pt"
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"Meilleur modèle sauvegardé: {checkpoint_path}")
                
                # Sauvegarder également dans le répertoire des modèles pré-entraînés
                pretrained_path = config.PRETRAINED_DIR / "best_model.pt"
                torch.save(self.model.state_dict(), pretrained_path)
                
                # Sauvegarder les métadonnées du modèle
                metadata = {
                    "model_type": self.model.__class__.__name__,
                    "epoch": epoch + 1,
                    "val_loss": val_metrics["loss"],
                    "val_f1": val_metrics["f1"],
                    "train_loss": train_metrics["loss"],
                    "train_f1": train_metrics["f1"],
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                with open(self.checkpoint_dir / "best_model_metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=4)
            else:
                patience_counter += 1
            
            # Sauvegarder un point de contrôle à chaque époque
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "history": history
            }, checkpoint_path)
            
            # Early stopping
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping après {epoch+1} époques")
                break
        
        # Sauvegarder l'historique complet
        with open(self.checkpoint_dir / "training_history.json", 'w') as f:
            json.dump(history, f, indent=4)
        
        return history
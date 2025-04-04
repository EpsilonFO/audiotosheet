# Audio-to-Sheet-Music: Conversion de fichiers audio en partitions de piano
# Structure du projet

### src/visualization/sheet_renderer.py

"""Module pour le rendu de partitions à partir de fichiers MIDI."""

import os
import subprocess
import tempfile
from pathlib import Path
import music21 as m21


class SheetRenderer:
    """Classe pour générer des partitions à partir de fichiers MIDI."""
    
    def __init__(self):
        """Initialiser le générateur de partitions."""
        pass
    
    def render(self, midi_path, output_path):
        """
        Générer une partition à partir d'un fichier MIDI.
        
        Args:
            midi_path: Chemin vers le fichier MIDI
            output_path: Chemin de sortie pour la partition
        """
        # Charger le fichier MIDI avec music21
        try:
            midi = m21.converter.parse(midi_path)
        except Exception as e:
            print(f"Erreur lors du chargement du fichier MIDI: {e}")
            return False
        
        # Extraire la partie de piano
        piano_part = None
        for part in midi.parts:
            if 'Piano' in part.partName or part.getInstrument().instrumentName == 'Piano':
                piano_part = part
                break
        
        # Si aucune partie de piano n'est trouvée, utiliser la première partie
        if piano_part is None and len(midi.parts) > 0:
            piano_part = midi.parts[0]
        elif piano_part is None:
            # Si aucune partie n'est trouvée, utiliser tout le score
            piano_part = midi
        
        # Ajouter quelques métadonnées
        piano_part.insert(0, m21.metadata.Metadata())
        piano_part.metadata.title = Path(midi_path).stem
        
        # Ajouter des éléments musicaux supplémentaires
        # Déterminer la mesure (essayer d'estimer à partir des données)
        from music21 import meter
        estimated_ts = meter.bestTimeSignature(piano_part)
        if estimated_ts:
            piano_part.insert(0, estimated_ts)
        else:
            piano_part.insert(0, meter.TimeSignature('4/4'))
        
        # Ajouter une clé (par défaut Do majeur / La mineur)
        from music21 import key
        piano_part.insert(0, key.KeySignature(0))
        
        # Exporter vers le format souhaité
        output_ext = Path(output_path).suffix.lower()
        
        if output_ext == '.pdf':
            # Utiliser Lilypond pour le rendu PDF
            try:
                # Créer un fichier Lilypond temporaire
                with tempfile.NamedTemporaryFile(suffix='.ly', delete=False) as tmp:
                    tmp_path = tmp.name
                
                # Exporter vers Lilypond
                ly_data = m21.lily.translate.createLilypondCode(piano_part)
                with open(tmp_path, 'w') as f:
                    f.write(ly_data)
                
                # Compiler avec Lilypond
                output_dir = str(Path(output_path).parent)
                output_name = Path(output_path).stem
                subprocess.run(['lilypond', '-o', os.path.join(output_dir, output_name), tmp_path], 
                               check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Nettoyer
                os.unlink(tmp_path)
                
            except Exception as e:
                print(f"Erreur lors du rendu avec Lilypond: {e}")
                # Utiliser MusicXML comme solution de repli
                midi.write('musicxml.pdf', fp=output_path)
        
        elif output_ext == '.png':
            # Utiliser le rendu PNG intégré
            midi.write('musicxml.png', fp=output_path)
        
        elif output_ext == '.xml' or output_ext == '.musicxml':
            # Exporter au format MusicXML
            midi.write('musicxml', fp=output_path)
        
        else:
            # Par défaut, exporter au format PDF
            midi.write('musicxml.pdf', fp=output_path)
        
        return True


### src/training/dataset.py
"""Module définissant le dataset pour l'entraînement."""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class AudioToMIDIDataset(Dataset):
    """Dataset pour l'entraînement de modèles de transcription audio vers MIDI."""
    
    def __init__(self, data_dir, num_frames=None, augmentation=False):
        """
        Initialiser le dataset.
        
        Args:
            data_dir: Répertoire contenant les fichiers de caractéristiques
            num_frames: Nombre de trames à utiliser (None pour utiliser toutes les trames)
            augmentation: Appliquer l'augmentation de données
        """
        self.data_dir = Path(data_dir)
        self.num_frames = num_frames
        self.augmentation = augmentation
        
        # Charger la liste des fichiers de caractéristiques
        self.file_list = list(self.data_dir.glob("*.json"))
        
        # S'assurer qu'il y a des fichiers
        if len(self.file_list) == 0:
            raise ValueError(f"Aucun fichier de caractéristiques trouvé dans {data_dir}")
        
        # Charger les annotations si elles existent dans un sous-répertoire "annotations"
        self.annotations_dir = self.data_dir.parent / "annotations"
        self.has_annotations = self.annotations_dir.exists()
        
    def __len__(self):
        """Retourner le nombre d'échantillons dans le dataset."""
        return len(self.file_list)
    
    def __getitem__(self, idx):
        """
        Récupérer un échantillon du dataset.
        
        Args:
            idx: Indice de l'échantillon
            
        Returns:
            tuple: (caractéristiques, cibles)
        """
        # Charger le fichier de caractéristiques
        file_path = self.file_list[idx]
        with open(file_path, 'r') as f:
            features_data = json.load(f)
        
        # Extraire le spectrogramme mel
        mel_spec = np.array(features_data["mel_spectrogram"])
        
        # Couper ou remplir pour obtenir un nombre fixe de trames si spécifié
        if self.num_frames is not None:
            if mel_spec.shape[1] > self.num_frames:
                # Couper
                start = np.random.randint(0, mel_spec.shape[1] - self.num_frames) if self.augmentation else 0
                mel_spec = mel_spec[:, start:start + self.num_frames]
            elif mel_spec.shape[1] < self.num_frames:
                # Remplir avec des zéros
                padding = np.zeros((mel_spec.shape[0], self.num_frames - mel_spec.shape[1]))
                mel_spec = np.concatenate([mel_spec, padding], axis=1)
        
        # Augmentation des données (si activée)
        if self.augmentation:
            # Ajouter un bruit aléatoire
            noise_level = np.random.uniform(0, 0.05)
            mel_spec = mel_spec + noise_level * np.random.randn(*mel_spec.shape)
            mel_spec = np.clip(mel_spec, 0, 1)  # Garder dans [0, 1]
            
            # Légères variations de tempo (time stretching)
            # Simulé en sautant ou dupliquant quelques trames
            if np.random.random() < 0.5 and mel_spec.shape[1] > 10:
                stretch_factor = np.random.uniform(0.9, 1.1)
                new_length = int(mel_spec.shape[1] * stretch_factor)
                indices = np.linspace(0, mel_spec.shape[1] - 1, new_length).astype(int)
                mel_spec = mel_spec[:, indices]
                
                # S'assurer que la taille est correcte si num_frames est spécifié
                if self.num_frames is not None:
                    if mel_spec.shape[1] > self.num_frames:
                        mel_spec = mel_spec[:, :self.num_frames]
                    elif mel_spec.shape[1] < self.num_frames:
                        padding = np.zeros((mel_spec.shape[0], self.num_frames - mel_spec.shape[1]))
                        mel_spec = np.concatenate([mel_spec, padding], axis=1)
        
        # Convertir en tensor
        mel_spec_tensor = torch.from_numpy(mel_spec).float()
        
        # Ajouter la dimension de canal
        mel_spec_tensor = mel_spec_tensor.unsqueeze(0)
        
        # Charger les annotations si disponibles
        if self.has_annotations:
            annotation_path = self.annotations_dir / f"{file_path.stem}.json"
            if annotation_path.exists():
                with open(annotation_path, 'r') as f:
                    annotation_data = json.load(f)
                
                # Convertir les annotations en matrice piano roll (frames x 88 notes)
                num_frames = mel_spec.shape[1]
                piano_roll = np.zeros((num_frames, 88))  # 88 notes de piano
                
                # Note MIDI 21 = A0, première note du piano
                for note_event in annotation_data["notes"]:
                    midi_note = note_event["midi_note"]
                    start_frame = int(note_event["start_time"] * features_data["sample_rate"] / config.HOP_LENGTH)
                    end_frame = int(note_event["end_time"] * features_data["sample_rate"] / config.HOP_LENGTH)
                    
                    # S'assurer que les frames sont dans les limites
                    start_frame = max(0, min(start_frame, num_frames - 1))
                    end_frame = max(0, min(end_frame, num_frames))
                    
                    # Remplir le piano roll
                    if midi_note >= 21 and midi_note <= 108:  # Vérifier que la note est dans la gamme du piano
                        piano_roll[start_frame:end_frame, midi_note - 21] = 1
                
                # Couper ou remplir pour correspondre au spectrogramme
                if piano_roll.shape[0] > mel_spec.shape[1]:
                    piano_roll = piano_roll[:mel_spec.shape[1], :]
                elif piano_roll.shape[0] < mel_spec.shape[1]:
                    padding = np.zeros((mel_spec.shape[1] - piano_roll.shape[0], piano_roll.shape[1]))
                    piano_roll = np.concatenate([piano_roll, padding], axis=0)
                
                # Convertir en tensor
                target_tensor = torch.from_numpy(piano_roll).float()
            else:
                # Si pas d'annotation disponible, créer une cible vide
                target_tensor = torch.zeros((mel_spec.shape[1], 88)).float()
        else:
            # Si pas d'annotations du tout, créer une cible vide
            target_tensor = torch.zeros((mel_spec.shape[1], 88)).float()
        
        return mel_spec_tensor, target_tensor


### src/training/metrics.py
"""Module pour calculer les métriques d'évaluation."""

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def calculate_metrics(predictions, targets, threshold=0.5):
    """
    Calculer les métriques d'évaluation pour la transcription de musique.
    
    Args:
        predictions: Prédictions du modèle (tensor)
        targets: Cibles réelles (tensor)
        threshold: Seuil pour binariser les prédictions
        
    Returns:
        dict: Métriques calculées (precision, recall, f1, accuracy)
    """
    # Convertir en numpy si nécessaire
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.numpy()
    
    # Binariser les prédictions
    binary_preds = (predictions > threshold).astype(np.int32)
    binary_targets = (targets > threshold).astype(np.int32)
    
    # Aplatir pour le calcul des métriques
    flat_preds = binary_preds.flatten()
    flat_targets = binary_targets.flatten()
    
    # Calculer précision, rappel et F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(
        flat_targets, flat_preds, average='binary', zero_division=0
    )
    
    # Calculer l'exactitude
    accuracy = accuracy_score(flat_targets, flat_preds)
    
    # Calculer les métriques spécifiques à la transcription de musique
    # Frame-level : évaluer la précision à chaque trame temporelle
    frame_metrics = {
        "precision": [],
        "recall": [],
        "f1": []
    }
    
    for i in range(binary_preds.shape[0]):  # Pour chaque batch
        for j in range(binary_preds.shape[1]):  # Pour chaque trame
            p, r, f, _ = precision_recall_fscore_support(
                binary_targets[i, j], binary_preds[i, j], average='binary', zero_division=0
            )
            frame_metrics["precision"].append(p)
            frame_metrics["recall"].append(r)
            frame_metrics["f1"].append(f)
    
    # Calculer les moyennes des métriques par trame
    frame_precision = np.mean(frame_metrics["precision"]) if frame_metrics["precision"] else 0
    frame_recall = np.mean(frame_metrics["recall"]) if frame_metrics["recall"] else 0
    frame_f1 = np.mean(frame_metrics["f1"]) if frame_metrics["f1"] else 0
    
    # Retourner toutes les métriques
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "frame_precision": frame_precision,
        "frame_recall": frame_recall,
        "frame_f1": frame_f1
    }


### src/training/trainer.py
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

# Fichiers principaux
```

## config.py
"""Configuration globale du projet."""

import os
from pathlib import Path

# Chemins
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MIDI_DIR = DATA_DIR / "midi"
MODELS_DIR = ROOT_DIR / "models"
PRETRAINED_DIR = MODELS_DIR / "pretrained"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

# Paramètres audio
SAMPLE_RATE = 44100
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
FMIN = 27.5  # A0 (la plus basse note du piano)
FMAX = 4186.0  # C8 (la plus haute note du piano)

# Paramètres du modèle
MODEL_TYPE = "cnn_lstm"  # Options: "cnn_lstm", "transformer"
DEVICE = "cuda"  # Options: "cpu", "cuda"

# Paramètres d'entraînement
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5

# Paramètres d'inférence
THRESHOLD = 0.5  # Seuil de détection des notes
MIN_NOTE_DURATION = 0.05  # Durée minimale d'une note en secondes

# Paramètres de visualisation
USE_LILYPOND = True  # Utiliser Lilypond pour le rendu de partition
```

## run.py
```python
"""Point d'entrée principal du projet."""

import argparse
import os
import sys
from pathlib import Path

import torch

# Ajouter le répertoire racine au PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing.audio_processor import AudioProcessor
from src.models.cnn_lstm import CNNLSTM
from src.models.transformer import TransformerModel
from src.training.trainer import Trainer
from src.inference.predictor import Predictor
from src.visualization.midi_generator import MIDIGenerator
from src.visualization.sheet_renderer import SheetRenderer
import config


def main():
    parser = argparse.ArgumentParser(description="Audio-to-Sheet-Music: Conversion de fichiers audio en partitions de piano")
    subparsers = parser.add_subparsers(dest="command", help="Commande à exécuter")

    # Sous-parseur pour la commande "preprocess"
    preprocess_parser = subparsers.add_parser("preprocess", help="Prétraiter les fichiers audio")
    preprocess_parser.add_argument("--input_dir", type=str, default=str(config.RAW_DATA_DIR), 
                                  help="Répertoire contenant les fichiers audio bruts")
    preprocess_parser.add_argument("--output_dir", type=str, default=str(config.PROCESSED_DATA_DIR),
                                  help="Répertoire de sortie pour les données prétraitées")

    # Sous-parseur pour la commande "train"
    train_parser = subparsers.add_parser("train", help="Entraîner le modèle")
    train_parser.add_argument("--data_dir", type=str, default=str(config.PROCESSED_DATA_DIR),
                             help="Répertoire contenant les données prétraitées")
    train_parser.add_argument("--model_type", type=str, default=config.MODEL_TYPE,
                             choices=["cnn_lstm", "transformer"], help="Type de modèle à utiliser")
    train_parser.add_argument("--checkpoint_dir", type=str, default=str(config.CHECKPOINTS_DIR),
                             help="Répertoire pour sauvegarder les points de contrôle")
    train_parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE,
                             help="Taille du batch")
    train_parser.add_argument("--lr", type=float, default=config.LEARNING_RATE,
                             help="Taux d'apprentissage")
    train_parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS,
                             help="Nombre d'époques d'entraînement")
    train_parser.add_argument("--device", type=str, default=config.DEVICE,
                             help="Appareil d'entraînement (cpu ou cuda)")

    # Sous-parseur pour la commande "convert"
    convert_parser = subparsers.add_parser("convert", help="Convertir un fichier audio en partition")
    convert_parser.add_argument("--input", type=str, required=True,
                               help="Chemin vers le fichier audio à convertir (.wav ou .mp3)")
    convert_parser.add_argument("--model_path", type=str, 
                               default=str(config.PRETRAINED_DIR / "best_model.pt"),
                               help="Chemin vers le modèle entraîné")
    convert_parser.add_argument("--model_type", type=str, default=config.MODEL_TYPE,
                               choices=["cnn_lstm", "transformer"], help="Type de modèle à utiliser")
    convert_parser.add_argument("--output_midi", type=str, default=None,
                               help="Chemin de sortie pour le fichier MIDI (par défaut: même nom que l'entrée)")
    convert_parser.add_argument("--output_sheet", type=str, default=None,
                               help="Chemin de sortie pour la partition (par défaut: même nom que l'entrée)")
    convert_parser.add_argument("--device", type=str, default=config.DEVICE,
                               help="Appareil d'inférence (cpu ou cuda)")

    args = parser.parse_args()

    # Créer les répertoires nécessaires s'ils n'existent pas
    for dir_path in [config.RAW_DATA_DIR, config.PROCESSED_DATA_DIR, config.MIDI_DIR,
                     config.PRETRAINED_DIR, config.CHECKPOINTS_DIR]:
        os.makedirs(dir_path, exist_ok=True)

    if args.command == "preprocess":
        processor = AudioProcessor(sample_rate=config.SAMPLE_RATE, 
                                  n_fft=config.N_FFT, 
                                  hop_length=config.HOP_LENGTH,
                                  n_mels=config.N_MELS,
                                  fmin=config.FMIN,
                                  fmax=config.FMAX)
        processor.process_directory(input_dir=args.input_dir, output_dir=args.output_dir)
        print(f"Prétraitement terminé. Résultats sauvegardés dans {args.output_dir}")

    elif args.command == "train":
        # Initialiser le modèle en fonction du type spécifié
        device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
        
        if args.model_type == "cnn_lstm":
            model = CNNLSTM(
                input_features=config.N_MELS,
                hidden_size=256,
                num_layers=2,
                num_notes=88  # 88 touches de piano
            ).to(device)
        else:  # transformer
            model = TransformerModel(
                input_features=config.N_MELS,
                d_model=512,
                nhead=8,
                num_encoder_layers=6,
                num_notes=88
            ).to(device)
        
        trainer = Trainer(
            model=model,
            data_dir=args.data_dir,
            checkpoint_dir=args.checkpoint_dir,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_epochs=args.epochs,
            device=device
        )
        
        trainer.train()
        print(f"Entraînement terminé. Modèle sauvegardé dans {args.checkpoint_dir}")

    elif args.command == "convert":
        # Charger le modèle
        device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
        
        if args.model_type == "cnn_lstm":
            model = CNNLSTM(
                input_features=config.N_MELS,
                hidden_size=256,
                num_layers=2,
                num_notes=88
            ).to(device)
        else:  # transformer
            model = TransformerModel(
                input_features=config.N_MELS,
                d_model=512,
                nhead=8,
                num_encoder_layers=6,
                num_notes=88
            ).to(device)
        
        # Vérifier si le modèle existe
        if not os.path.exists(args.model_path):
            print(f"Erreur: Le modèle spécifié ({args.model_path}) n'existe pas.")
            print("Veuillez d'abord entraîner un modèle ou spécifier un modèle pré-entraîné valide.")
            return

        # Charger les poids du modèle
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        
        # Initialiser le prédicateur
        predictor = Predictor(
            model=model,
            sample_rate=config.SAMPLE_RATE,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS,
            fmin=config.FMIN,
            fmax=config.FMAX,
            threshold=config.THRESHOLD,
            device=device
        )
        
        # Préparer les chemins de sortie
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Erreur: Le fichier d'entrée ({args.input}) n'existe pas.")
            return
            
        if args.output_midi is None:
            output_midi = config.MIDI_DIR / f"{input_path.stem}.mid"
        else:
            output_midi = Path(args.output_midi)
            
        if args.output_sheet is None:
            output_sheet = config.MIDI_DIR / f"{input_path.stem}.pdf"
        else:
            output_sheet = Path(args.output_sheet)
            
        # Convertir l'audio en prédictions de notes
        print(f"Conversion de {args.input}...")
        note_events = predictor.predict(args.input)
        
        # Générer le fichier MIDI
        midi_generator = MIDIGenerator(min_note_duration=config.MIN_NOTE_DURATION)
        midi_generator.generate(note_events, output_path=str(output_midi))
        print(f"Fichier MIDI généré: {output_midi}")
        
        # Générer la partition si Lilypond est activé
        if config.USE_LILYPOND:
            sheet_renderer = SheetRenderer()
            sheet_renderer.render(midi_path=str(output_midi), output_path=str(output_sheet))
            print(f"Partition générée: {output_sheet}")
        
        print("Conversion terminée avec succès!")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
```

## Implémentation des modules principaux

### src/preprocessing/audio_processor.py
```python
"""Module de prétraitement des fichiers audio."""

import os
import glob
import json
import numpy as np
import librosa
import torch
import torchaudio
from pathlib import Path


class AudioProcessor:
    """Classe pour prétraiter les fichiers audio et extraire des caractéristiques."""

    def __init__(self, sample_rate=44100, n_fft=2048, hop_length=512, n_mels=128, 
                 fmin=27.5, fmax=4186.0):
        """
        Initialiser le processeur audio.
        
        Args:
            sample_rate: Taux d'échantillonnage cible
            n_fft: Taille de la fenêtre FFT
            hop_length: Nombre d'échantillons entre les trames successives
            n_mels: Nombre de bandes mel
            fmin: Fréquence minimale (Hz)
            fmax: Fréquence maximale (Hz)
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
    
    def process_file(self, file_path, output_dir=None):
        """
        Prétraiter un fichier audio et extraire des caractéristiques.
        
        Args:
            file_path: Chemin vers le fichier audio
            output_dir: Répertoire de sortie pour les caractéristiques
        
        Returns:
            dict: Caractéristiques extraites
        """
        # Charger le fichier audio
        try:
            y, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
        except Exception as e:
            print(f"Erreur lors du chargement de {file_path}: {e}")
            return None
        
        # Extraire le spectrogramme mel
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        
        # Convertir en dB avec un plancher à -80dB
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=80.0)
        
        # Normaliser entre 0 et 1
        mel_spec_norm = (mel_spec_db + 80.0) / 80.0
        
        # Extraire des caractéristiques supplémentaires
        chroma = librosa.feature.chroma_stft(
            y=y, 
            sr=sr, 
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        onset_env = librosa.onset.onset_strength(
            y=y, 
            sr=sr,
            hop_length=self.hop_length
        )
        
        # Détection des débuts de notes
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=self.hop_length,
            backtrack=True
        )
        
        onset_times = librosa.frames_to_time(onsets, sr=sr, hop_length=self.hop_length)
        
        # Créer un dictionnaire de caractéristiques
        features = {
            "file_name": os.path.basename(file_path),
            "sample_rate": sr,
            "duration": librosa.get_duration(y=y, sr=sr),
            "mel_spectrogram": mel_spec_norm.tolist(),
            "chroma": chroma.tolist(),
            "onset_times": onset_times.tolist()
        }
        
        # Sauvegarder les caractéristiques si un répertoire de sortie est spécifié
        if output_dir is not None:
            output_path = Path(output_dir) / f"{Path(file_path).stem}.json"
            with open(output_path, 'w') as f:
                json.dump(features, f)
        
        return features
    
    def process_directory(self, input_dir, output_dir):
        """
        Prétraiter tous les fichiers audio d'un répertoire.
        
        Args:
            input_dir: Répertoire contenant les fichiers audio
            output_dir: Répertoire de sortie pour les caractéristiques
        """
        # Créer le répertoire de sortie s'il n'existe pas
        os.makedirs(output_dir, exist_ok=True)
        
        # Trouver tous les fichiers audio
        audio_files = []
        for ext in ['*.wav', '*.mp3']:
            audio_files.extend(glob.glob(os.path.join(input_dir, ext)))
        
        print(f"Traitement de {len(audio_files)} fichiers audio...")
        
        # Traiter chaque fichier
        for i, file_path in enumerate(audio_files):
            print(f"[{i+1}/{len(audio_files)}] Traitement de {os.path.basename(file_path)}...")
            self.process_file(file_path, output_dir)
        
        print("Traitement terminé.")
```

### src/models/cnn_lstm.py
```python
"""Modèle hybride CNN-LSTM pour la transcription de musique."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLSTM(nn.Module):
    """
    Modèle hybride CNN-LSTM pour la transcription de musique.
    
    Architecture:
        1. Couches CNN pour extraire les caractéristiques locales
        2. BiLSTM pour capturer les dépendances temporelles
        3. Couche entièrement connectée pour la prédiction des notes
    """
    
    def __init__(self, input_features=128, hidden_size=256, num_layers=2, num_notes=88):
        """
        Initialiser le modèle CNN-LSTM.
        
        Args:
            input_features: Nombre de caractéristiques d'entrée (bandes mel)
            hidden_size: Taille de l'état caché LSTM
            num_layers: Nombre de couches LSTM
            num_notes: Nombre de notes à prédire (88 pour le piano standard)
        """
        super(CNNLSTM, self).__init__()
        
        # Couches CNN
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout1 = nn.Dropout(0.25)
        
        # Calculer la taille après les couches CNN
        # Après 3 couches de pooling, les dimensions sont divisées par 2^3 = 8
        cnn_output_height = input_features // 8
        
        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=128 * cnn_output_height,  # Sortie de la dernière couche CNN
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        self.dropout2 = nn.Dropout(0.3)
        
        # Couche entièrement connectée
        self.fc = nn.Linear(hidden_size * 2, num_notes)  # *2 pour bidirectionnel
    
    def forward(self, x):
        """
        Propagation avant du modèle.
        
        Args:
            x: Tensor d'entrée de forme [batch_size, 1, num_frames, input_features]
                où num_frames est le nombre de trames temporelles
            
        Returns:
            Tensor de forme [batch_size, num_frames, num_notes]
                représentant la probabilité de chaque note à chaque trame
        """
        batch_size, _, num_frames, input_features = x.shape
        
        # Couches CNN
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout1(x)
        
        # Réorganiser pour le LSTM: [batch, channels, frames, features] -> [batch, frames, channels*features]
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batch_size, -1, x.shape[1] * x.shape[3])
        
        # BiLSTM
        x, _ = self.lstm(x)
        x = self.dropout2(x)
        
        # Couche entièrement connectée
        x = self.fc(x)
        
        # Activation sigmoid pour la prédiction multi-étiquettes
        x = torch.sigmoid(x)
        
        return x
```

### src/inference/predictor.py
```python
"""Module pour l'inférence et la prédiction à partir d'audio."""

import torch
import librosa
import numpy as np
from pathlib import Path


class Predictor:
    """Classe pour prédire les notes de piano à partir d'un fichier audio."""
    
    def __init__(self, model, sample_rate=44100, n_fft=2048, hop_length=512, 
                 n_mels=128, fmin=27.5, fmax=4186.0, threshold=0.5, device="cpu"):
        """
        Initialiser le prédicateur.
        
        Args:
            model: Modèle de deep learning entraîné
            sample_rate: Taux d'échantillonnage
            n_fft: Taille de la fenêtre FFT
            hop_length: Nombre d'échantillons entre les trames successives
            n_mels: Nombre de bandes mel
            fmin: Fréquence minimale (Hz)
            fmax: Fréquence maximale (Hz)
            threshold: Seuil de probabilité pour la détection des notes
            device: Appareil pour l'inférence ("cpu" ou "cuda")
        """
        self.model = model
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.threshold = threshold
        self.device = device
        
        # Créer une correspondance entre les indices et les notes MIDI
        # Note MIDI 21 = A0, première note du piano
        self.note_range = range(21, 109)  # 88 notes
    
    def preprocess_audio(self, audio_path):
        """
        Prétraiter un fichier audio pour l'inférence.
        
        Args:
            audio_path: Chemin vers le fichier audio
            
        Returns:
            Tensor prétraité prêt pour l'inférence
        """
        # Charger le fichier audio
        y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        
        # Extraire le spectrogramme mel
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        
        # Convertir en dB avec un plancher à -80dB
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=80.0)
        
        # Normaliser entre 0 et 1
        mel_spec_norm = (mel_spec_db + 80.0) / 80.0
        
        # Convertir en tensor et ajouter les dimensions de batch et de canal
        x = torch.from_numpy(mel_spec_norm).float()
        x = x.unsqueeze(0).unsqueeze(0)  # [batch=1, channel=1, num_mels, num_frames]
        
        return x
    
    def predict(self, audio_path):
        """
        Prédire les notes de piano à partir d'un fichier audio.
        
        Args:
            audio_path: Chemin vers le fichier audio
            
        Returns:
            Liste d'événements de notes (note, temps_début, temps_fin)
        """
        # Prétraiter l'audio
        x = self.preprocess_audio(audio_path)
        x = x.to(self.device)
        
        # Prédire les probabilités de notes
        with torch.no_grad():
            predictions = self.model(x)
        
        # Convertir en numpy
        predictions = predictions.squeeze(0).cpu().numpy()  # [num_frames, num_notes]
        
        # Appliquer le seuil pour détecter les notes actives
        note_activations = predictions > self.threshold
        
        # Convertir en événements de notes
        note_events = []
        
        # Pour chaque note (88 notes de piano)
        for note_idx in range(note_activations.shape[1]):
            # Trouver les segments où la note est active
            frames = note_activations[:, note_idx]
            
            # Trouver les changements d'état (0->1 ou 1->0)
            changes = np.diff(frames.astype(int), prepend=0, append=0)
            
            # Indices où la note commence (0->1)
            note_starts = np.where(changes == 1)[0]
            
            # Indices où la note se termine (1->0)
            note_ends = np.where(changes == -1)[0]
            
            # Créer des événements de notes
            for start, end in zip(note_starts, note_ends):
                midi_note = self.note_range[note_idx]
                start_time = librosa.frames_to_time(start, sr=self.sample_rate, hop_length=self.hop_length)
                end_time = librosa.frames_to_time(end, sr=self.sample_rate, hop_length=self.hop_length)
                
                note_events.append({
                    'note': midi_note,
                    'start_time': start_time,
                    'end_time': end_time
                })
        
        # Trier les événements par temps de début
        note_events.sort(key=lambda x: x['start_time'])
        
        return note_events
```

### src/visualization/midi_generator.py
```python
"""Module pour la génération de fichiers MIDI à partir de prédictions."""

import mido
import numpy as np


class MIDIGenerator:
    """Classe pour générer des fichiers MIDI à partir d'événements de notes."""
    
    def __init__(self, min_note_duration=0.05, velocity=64, ticks_per_beat=480):
        """
        Initialiser le générateur MIDI.
        
        Args:
            min_note_duration: Durée minimale d'une note en secondes
            velocity: Vélocité des notes (0-127)
            ticks_per_beat: Résolution temporelle du MIDI
        """
        self.min_note_duration = min_note_duration
        self.velocity = velocity
        self.ticks_per_beat = ticks_per_beat
        
    def generate(self, note_events, output_path, tempo=500000):  # tempo en microsecondes par beat (500000 = 120 BPM)
        """
        Générer un fichier MIDI à partir d'événements de notes.
        
        Args:
            note_events: Liste d'événements de notes (note, temps_début, temps_fin)
            output_path: Chemin de sortie pour le fichier MIDI
            tempo: Tempo en microsecondes par beat
        """
        # Créer un nouveau fichier MIDI
        mid = mido.MidiFile(ticks_per_beat=self.ticks_per_beat)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        
        # Ajouter un message de tempo
        track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))
        
        # Ajouter un message d'instrument (0 = Piano à queue)
        track.append(mido.Message('program_change', program=0, time=0))
        
        # Filtrer les notes trop courtes
        filtered_events = [event for event in note_events 
                           if event['end_time'] - event['start_time'] >= self.min_note_duration]
        
        # Convertir les temps en ticks
        seconds_per_tick = (tempo / 1000000) / self.ticks_per_beat
        
        last_event_time = 0
        
        # Trier les événements par temps de début
        events_sorted = sorted(filtered_events, key=lambda x: x['start_time'])
        
        # Ajouter les événements de note
        for event in events_sorted:
            note = event['note']
            start_time = event['start_time']
            end_time = event['end_time']
            
            # Calculer le delta time pour le message note_on
            delta_time_on = int((start_time - last_event_time) / seconds_per_tick)
            
            # Ajouter le message note_on
            track.append(mido.Message('note_on', note=note, velocity=self.velocity, time=delta_time_on))
            
            # Mettre à jour le dernier temps d'événement
            last_event_time = start_time
            
            # Calculer le delta time pour le message note_off
            delta_time_off = int((end_time - start_time) / seconds_per_tick)
            
            # Ajouter le message note_off
            track.append(mido.Message('note_off', note=note, velocity=0, time=delta_time_off))
            
            # Mettre à jour le dernier temps d'événement
            last_event_time = end_time
        
        # Sauvegarder le fichier MIDI
        mid.save(output_path)
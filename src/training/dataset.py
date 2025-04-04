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

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
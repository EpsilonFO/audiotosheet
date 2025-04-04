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
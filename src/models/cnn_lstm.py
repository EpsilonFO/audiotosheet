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
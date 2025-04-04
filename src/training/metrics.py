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
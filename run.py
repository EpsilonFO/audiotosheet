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
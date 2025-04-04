
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
                
                # Nettoyerr
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
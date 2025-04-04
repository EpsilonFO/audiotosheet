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
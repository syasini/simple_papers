import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import streamlit as st
from loguru import logger

from simple_papers.path_handler import PathHandler


class AudioHandler:
    """Class to handle audio generation and management for paper summaries."""
    
    def __init__(self, paper_path: str, default_audio_voice: str = "Alloy"):
        """
        Initialize an AudioHandler object.
        
        Parameters
        ----------
        paper_path : str
            Path to the paper file
        """
        self.paper_path = paper_path
        self.path_handler = PathHandler(paper_path)
        self.audio_mapping_path = self.path_handler.audio_mapping_path
        self.audio_dir = self.path_handler.audio_dir

        self._default_audio_voice = default_audio_voice
        
        # Ensure audio directory exists
        self.audio_dir.mkdir(exist_ok=True)
        
        # Load audio mapping if it exists
        self._audio_mapping = self._load_audio_mapping()

        # Map section types to voices
        # This uses the updated voice mapping approach from utils.py
        self._voice_mapping = {
            "title": "Football",  # Enthusiastic voice for titles
            "else": self._default_audio_voice,  # Default voice for other sections
        }

    @property
    def default_audio_voice(self):
        return self._default_audio_voice

    @default_audio_voice.setter
    def default_audio_voice(self, value):
        self._default_audio_voice = value
        self._voice_mapping["else"] = value

    @property
    def voice_mapping(self):
        return self._voice_mapping
    
    def _load_audio_mapping(self) -> Dict[str, list]:
        """
        Load audio mapping from file if it exists.
        
        Returns
        -------
        Dict[str, list]
            Dictionary mapping section IDs to lists of audio file paths
        """
        if self.audio_mapping_path.exists():
            try:
                with open(self.audio_mapping_path, "r") as f:
                    mapping = json.load(f)
                # Convert any string values to lists for backward compatibility
                for key, value in mapping.items():
                    if isinstance(value, str):
                        mapping[key] = [value]
                return mapping
            except Exception as e:
                logger.error(f"Error loading audio mapping: {str(e)}")
                return {}
        return {}
    
    def _save_audio_mapping(self) -> None:
        """Save audio mapping to file."""
        try:
            with open(self.audio_mapping_path, "w") as f:
                json.dump(self._audio_mapping, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving audio mapping: {str(e)}")

    def delete_audio_file(self, group_id: str, voice: str = None) -> None:
        """
        Delete audio file for a section.
        
        Parameters
        ----------
        group_id : str
            ID of the section to delete audio for
        voice : str, optional
            Voice name to delete specifically. If None, deletes all audio files for the group_id
        """
        if group_id in self._audio_mapping:
            if voice is None:
                # Delete all audio files for this group_id
                for file_path_str in self._audio_mapping[group_id]:
                    file_path = Path(file_path_str)
                    if file_path.exists():
                        try:
                            logger.info(f"Deleting audio file for {group_id} from {file_path}")
                            file_path.unlink()  # Delete the file from the filesystem
                        except Exception as e:
                            logger.error(f"Error deleting audio file {file_path}: {str(e)}")
                del self._audio_mapping[group_id]
            else:
                # Delete only audio files for this group_id with the specified voice
                files_to_keep = []
                for file_path_str in self._audio_mapping[group_id]:
                    file_path = Path(file_path_str)
                    if voice in file_path_str:
                        if file_path.exists():
                            try:
                                logger.info(f"Deleting audio file for {group_id} with voice {voice} from {file_path}")
                                file_path.unlink()  # Delete the file from the filesystem
                            except Exception as e:
                                logger.error(f"Error deleting audio file {file_path}: {str(e)}")
                    else:
                        files_to_keep.append(file_path_str)
                if files_to_keep:
                    self._audio_mapping[group_id] = files_to_keep
                else:
                    del self._audio_mapping[group_id]
            self._save_audio_mapping()
        
    def save_summary_audio_to_file(self, group_id: str, summary: str) -> str:
        """
        Save summary text as audio to a file.
        
        Parameters
        ----------
        group_id : str
            ID of the section
        summary : str
            Summary text to convert to audio
            
        Returns
        -------
        str
            Path to the saved audio file
        """
        from simple_papers.utils import text_to_audio_file
        
        # Determine voice based on section type
        voice_name = self.voice_mapping["title"] if "title" in group_id else self.voice_mapping["else"]
        
        # Create filename based on section ID and voice
        paper_name = self.path_handler.name
        filename = f"{paper_name}_{group_id}_{voice_name}.mp3"
        file_path = str(self.audio_dir / filename)
        
        # Check if file already exists
        if os.path.exists(file_path):
            logger.info(f"Audio file for {group_id} already exists at {file_path}")
            
            # Make sure it's in the mapping
            if group_id not in self._audio_mapping:
                self._audio_mapping[group_id] = [file_path]
                self._save_audio_mapping()
            elif file_path not in self._audio_mapping[group_id]:
                self._audio_mapping[group_id].append(file_path)
                self._save_audio_mapping()
                
            return file_path
        
        try:
            # Generate audio and save to file in one step
            logger.info(f"Converting summary for {group_id} to audio")
            saved_path = text_to_audio_file(summary, file_path, voice_name)
                
            # Update mapping
            if group_id not in self._audio_mapping:
                self._audio_mapping[group_id] = [saved_path]
            elif saved_path not in self._audio_mapping[group_id]:
                self._audio_mapping[group_id].append(saved_path)
            self._save_audio_mapping()
            
            logger.info(f"Saved audio for {group_id} to {saved_path}")
            return saved_path
        except Exception as e:
            logger.error(f"Error saving audio for {group_id}: {str(e)}")
            return ""
    
    def get_audio_bytes(self, group_id: str, summary: str = None, voice: str = None) -> bytes:
        """
        Get audio bytes for a section, generating the file if needed.
        
        Parameters
        ----------
        group_id : str
            The section ID
        summary : str, optional
            Summary text to use if audio file needs to be generated
        voice : str, optional
            Voice to use for audio file selection
            
        Returns
        -------
        bytes
            Audio bytes for the section
        """
        # Check if we have a cached audio file
        if group_id in self._audio_mapping and self._audio_mapping[group_id]:
            # If voice is specified, try to find an audio file with that voice
            if voice:
                for file_path_str in self._audio_mapping[group_id]:
                    if voice in file_path_str:
                        file_path = Path(file_path_str)
                        if file_path.exists():
                            try:
                                logger.info(f"Using cached audio for {group_id} with voice {voice} from {file_path}")
                                with open(file_path, "rb") as f:
                                    return f.read()
                            except Exception as e:
                                logger.error(f"Error reading audio file {file_path}: {str(e)}")
            
            # If no voice specified or no matching file found, use the first available one
            file_path = Path(self._audio_mapping[group_id][0])
            if file_path.exists():
                try:
                    logger.info(f"Using cached audio for {group_id} from {file_path}")
                    with open(file_path, "rb") as f:
                        return f.read()
                except Exception as e:
                    logger.error(f"Error reading audio file {file_path}: {str(e)}")
        
        # Generate new audio if we don't have it or if there was an error
        if summary is not None:
            logger.info(f"No cached audio found for {group_id}, generating new audio")
            # If voice is specified, temporarily set it as default for generation
            original_voice = None
            if voice:
                original_voice = self._default_audio_voice
                self._default_audio_voice = voice
                
            file_path = self.save_summary_audio_to_file(group_id, summary)
            
            # Restore original voice if it was changed
            if original_voice:
                self._default_audio_voice = original_voice
                
            if file_path:
                try:
                    with open(file_path, "rb") as f:
                        return f.read()
                except Exception as e:
                    logger.error(f"Error reading newly generated audio file {file_path}: {str(e)}")
                    
        # Return empty bytes if all else fails
        logger.warning(f"Unable to get audio bytes for {group_id}, returning empty bytes")
        return b""
    
    def generate_all_audio(self, summaries: Dict[str, str], voice: str = None) -> Dict[str, List[str]]:
        """
        Generate audio files for all summaries.
        
        Parameters
        ----------
        summaries : Dict[str, str]
            Dictionary mapping section IDs to summary text
        voice : str, optional
            Voice to use for audio generation. If None, uses the default voice.
            
        Returns
        -------
        Dict[str, List[str]]
            Dictionary mapping section IDs to lists of audio file paths
        """
        results = {}
        failed = []
        
        # Create progress bar and status elements
        total = len(summaries)
        progress_bar = st.sidebar.progress(0.0)
        status_text = st.sidebar.empty()
        
        logger.info(f"Starting batch audio generation for {total} summaries")
        
        # Save original voice if specified voice is different
        original_voice = None
        if voice and voice != self._default_audio_voice:
            original_voice = self._default_audio_voice
            self._default_audio_voice = voice
            
        # Process each summary
        for i, (group_id, summary) in enumerate(summaries.items()):
            try:
                # Show current progress in UI
                status_text.text(f"Generating audio: {group_id} ({i+1}/{total})")
                
                # Generate the audio file
                file_path = self.save_summary_audio_to_file(group_id, summary)
                
                if file_path:
                    if group_id not in results:
                        results[group_id] = [file_path]
                    else:
                        results[group_id].append(file_path)
                else:
                    failed.append(group_id)
            except Exception as e:
                logger.error(f"Failed to generate audio for {group_id}: {str(e)}")
                failed.append(group_id)
            finally:
                # Update progress bar
                progress = (i + 1) / total
                progress_bar.progress(progress)
        
        # Show completion status
        success_count = len(results)
        if failed:
            status_text.text(f"Audio generation completed with {success_count}/{total} successful ({len(failed)} failed)")
            logger.warning(f"Failed to generate audio for sections: {', '.join(failed)}")
        else:
            status_text.text(f"Audio generation complete! {success_count}/{total} successful")
            logger.info("All audio files generated successfully")
        
        # Restore original voice if it was changed
        if original_voice:
            self._default_audio_voice = original_voice
            
        return results

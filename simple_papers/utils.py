
from pathlib import Path
import io
import streamlit as st
import os
from openai import OpenAI
from loguru import logger
from pypdf import PdfReader

# Voice mapping dictionary: name -> (engine, voice_id)
VOICE_MAPPING = {
    "Football": ("elevenlabs", "gU0LNdkMOQCOrPrwtbee"),  # Default ElevenLabs voice
    "Joe": ("elevenlabs", "ch0vU2DwfJVmFG2iZy89"),  # Another ElevenLabs voice
    "Felicity": ("elevenlabs", "aTbnroHRGIomiKpqAQR8"),
    "Amelia": ("elevenlabs", "ZF6FPAbjXT4488VcRRnw"),
    "Hope": ("elevenlabs", "uYXf8XasLslADfZ2MB4u"),

    "Alloy": ("openai", "alloy"),  # OpenAI voice
    "Echo": ("openai", "echo"),  # OpenAI voice
    "Fable": ("openai", "fable"),  # OpenAI voice
    "Onyx": ("openai", "onyx"),  # OpenAI voice
    "Nova": ("openai", "nova")  # OpenAI voice
}

def load_pdf_to_binary(file_path):
    """Load a PDF file and return its binary data."""
    try:
        with open(file_path, 'rb') as file:
            binary_data = file.read()
        return binary_data
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def get_page_dimensions(pdf_path):
    """Get the dimensions of the first page of a PDF"""
    reader = PdfReader(pdf_path)
    if reader.pages:
        # Get media box (page size)
        media_box = reader.pages[0].mediabox
        width = float(media_box.width)
        height = float(media_box.height)
        return width, height
    return 612, 792  # Default letter size if dimensions can't be determined


def get_page_dimensions_pymupdf(pdf_path):
    """Get the dimensions of the first page of a PDF using PyMuPDF"""
    import fitz  # PyMuPDF - this is the correct import name
    doc = fitz.open(pdf_path)
    if doc.page_count > 0:
        page = doc[0]
        width, height = page.rect.width, page.rect.height
        doc.close()
        return width, height
    doc.close()
    return 612, 792  # Default letter size

def get_n_pages(pdf):
    """Get the number of pages in a PDF.
    
    Args:
        pdf: Can be:
            - A file path (str)
            - Binary data (bytes) from load_pdf_to_binary
            - A Streamlit UploadedFile object
            - A file-like object with read/seek methods
    
    Returns:
        int: Number of pages in the PDF, or 0 if the PDF couldn't be processed.
    """
    from io import BytesIO
    
    try:
        # Handle None or empty bytes
        if pdf is None or (isinstance(pdf, bytes) and len(pdf) == 0):
            return 0
        
        # Handle Streamlit UploadedFile object
        if hasattr(pdf, 'read') and callable(pdf.read):
            pdf.seek(0)  # Reset file pointer
            reader = PdfReader(pdf)
        # Handle binary data from load_pdf_to_binary
        elif isinstance(pdf, bytes):
            reader = PdfReader(BytesIO(pdf))
        # Handle file path or other file-like object
        else:
            reader = PdfReader(pdf)
        
        return len(reader.pages)
    except Exception:
        # Silently return 0 if any errors occur
        return -1

def get_annotation_dimensions(l, t, r, b, paper_width, paper_height):
    """Convert normalized bounding box coordinates to annotation dimensions.
    
    Args:
        l: Left coordinate (normalized 0-1)
        t: Top coordinate (normalized 0-1)
        r: Right coordinate (normalized 0-1)
        b: Bottom coordinate (normalized 0-1)
        paper_width: Width of the paper in points
        paper_height: Height of the paper in points
        
    Returns:
        dict: Dictionary with x, y, width, and height keys for annotation
    """
    x = int(l * paper_width)
    y = int(t * paper_height)
    width = int((r - l) * paper_width)
    height = int((b - t) * paper_height)
    
    return {
        "x": x,
        "y": y,
        "width": width,
        "height": height
    }


# =======
# Audio Conversion
# =======

def get_elevenlabs_client():
    """
    Get an initialized ElevenLabs client.
    
    Returns
    -------
    ElevenLabs
        Initialized ElevenLabs client
    """
    from elevenlabs import ElevenLabs
    return ElevenLabs(api_key=st.secrets["ELEVENLABS_API_KEY"])


def get_openai_client():
    """
    Get an initialized OpenAI client.
    
    Returns
    -------
    OpenAI
        Initialized OpenAI client
    """
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


def convert_to_speech_openai(text: str, voice: str = "alloy", model: str = "tts-1-hd"):
    """
    Convert text to speech using OpenAI's TTS API.
    
    Parameters
    ----------
    text : str
        The text to convert to speech
    voice : str, default="alloy"
        OpenAI voice to use (alloy, echo, fable, onyx, nova)
    model : str, default="tts-1-hd"
        OpenAI TTS model to use
        
    Returns
    -------
    response
        Raw response from the OpenAI API
        
    Raises
    ------
    Exception
        If there's an error during audio generation
    """
    try:
        client = get_openai_client()
        audio_speed = 1.1
        logger.info(f"Generating audio with speed: {audio_speed}")
        response = client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            speed=audio_speed,
        )
        return response
    except Exception as e:
        st.error(f"Error with OpenAI TTS: {str(e)}")
        raise Exception(f"Error generating audio with OpenAI: {str(e)}")


def convert_to_speech_elevenlabs(text: str, voice_id: str = "0S5oIfi8zOZixuSj8K6n"):
    """
    Convert text to speech using ElevenLabs API.
    
    Parameters
    ----------
    text : str
        The text to convert to speech
    voice_id : str, default="0S5oIfi8zOZixuSj8K6n"
        ElevenLabs voice ID to use for speech synthesis
        
    Returns
    -------
    response
        Raw response from the ElevenLabs API
        
    Raises
    ------
    Exception
        If there's an error during audio generation
    """
    from elevenlabs import VoiceSettings
    
    try:
        # Get ElevenLabs client
        client = get_elevenlabs_client()
        audio_speed = 1.15
        logger.info(f"Generating audio with speed: {audio_speed}")
        # Generate audio from text with specific voice ID
        response = client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id="eleven_turbo_v2_5",
            output_format="mp3_44100_128",
            voice_settings=VoiceSettings(
                stability=0.5,
                similarity_boost=0.75,
                style=0.2,
                speed=audio_speed,
                use_speaker_boost=True
            )
        )
        
        return response
    except Exception as e:
        st.error(f"Error with ElevenLabs TTS: {str(e)}")
        raise Exception(f"Error generating audio with ElevenLabs: {str(e)}")


def save_audio_to_file(response, file_path: str, engine: str = "elevenlabs") -> str:
    """
    Save audio response to a file.
    
    Parameters
    ----------
    response
        Audio response from TTS API (ElevenLabs or OpenAI)
    file_path : str
        Path to save the audio file
    engine : str, default="elevenlabs"
        The TTS engine that generated the response ("elevenlabs" or "openai")
        
    Returns
    -------
    str
        Path to the saved audio file
        
    Raises
    ------
    Exception
        If there's an error during file saving
    """
    try:
        # Ensure directory exists
        Path(file_path).parent.mkdir(exist_ok=True, parents=True)
        
        # Write audio data to file
        with open(file_path, "wb") as f:
            if engine == "openai":
                # OpenAI response has a read() method
                f.write(response.read())
            else:  # elevenlabs
                # If response is a stream/iterator
                if hasattr(response, '__iter__') and not isinstance(response, (bytes, str)):
                    for chunk in response:
                        if chunk:
                            f.write(chunk)
                else:
                    # If response is already complete bytes object
                    f.write(response)
                
        return file_path
    except Exception as e:
        raise Exception(f"Error saving audio to file: {str(e)}")


def response_to_audio_bytes(response, engine: str = "elevenlabs"):
    """
    Convert TTS API response to audio bytes.
    
    Parameters
    ----------
    response
        Response from TTS API (ElevenLabs or OpenAI)
    engine : str, default="elevenlabs"
        The TTS engine that generated the response ("elevenlabs" or "openai")
        
    Returns
    -------
    bytes
        Audio data as bytes
        
    Raises
    ------
    Exception
        If there's an error during conversion
    """
    if engine == "openai":
        # OpenAI response has a read() method
        return response.read()
    
    # ElevenLabs handling
    if isinstance(response, bytes):
        return response
        
    # If response is a stream/iterator, read all chunks
    if hasattr(response, '__iter__'):
        audio_bytes = b""
        for chunk in response:
            if chunk:
                audio_bytes += chunk
        return audio_bytes
        
    # If response has a read method (like a file object)
    if hasattr(response, 'read'):
        return response.read()
        
    # If we can't determine the format, return as is
    return response


def get_audio_response(text: str, voice_name: str = "Joe"):
    """
    Get raw audio response from the appropriate TTS engine based on voice name.
    
    Parameters
    ----------
    text : str
        The text to convert to speech
    voice_name : str, default="Joe Davis"
        Name of the voice to use (must be in VOICE_MAPPING)
        
    Returns
    -------
    tuple
        (response, engine) - Raw response from the TTS API and the engine name
        
    Raises
    ------
    Exception
        If there's an error during audio generation
    """
    if voice_name not in VOICE_MAPPING:
        raise ValueError(f"Unknown voice: {voice_name}. Available voices: {list(VOICE_MAPPING.keys())}")
    
    engine, voice_id = VOICE_MAPPING[voice_name]
    
    if engine == "openai":
        return convert_to_speech_openai(text, voice_id), engine
    else:  # elevenlabs
        return convert_to_speech_elevenlabs(text, voice_id), engine


def text_to_audio_bytes(text: str, voice_name: str = "Joe") -> bytes:
    """
    Convert text to audio bytes using the appropriate TTS engine.
    
    Parameters
    ----------
    text : str
        The text to convert to speech
    voice_name : str, default="Joe Davis"
        Name of the voice to use (must be in VOICE_MAPPING)
        
    Returns
    -------
    bytes
        Audio data as bytes
        
    Raises
    ------
    Exception
        If there's an error during audio generation
    """
    response, engine = get_audio_response(text, voice_name)
    return response_to_audio_bytes(response, engine)


def text_to_audio_file(text: str, file_path: str, voice_name: str = "Joe") -> str:
    """
    Convert text to audio and save directly to a file using the appropriate TTS engine.
    
    Parameters
    ----------
    text : str
        The text to convert to speech
    file_path : str
        Path to save the audio file
    voice_name : str, default="Joe Davis"
        Name of the voice to use (must be in VOICE_MAPPING)
        
    Returns
    -------
    str
        Path to the saved audio file
        
    Raises
    ------
    Exception
        If there's an error during audio generation or saving
    """
    response, engine = get_audio_response(text, voice_name)
    return save_audio_to_file(response, file_path, engine)



from pathlib import Path
import io
import streamlit as st

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


def get_audio_response(text: str, voice_id: str = "0S5oIfi8zOZixuSj8K6n"):
    """
    Get raw audio response from ElevenLabs API.
    
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
        
        # Generate audio from text with specific voice ID
        response = client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id="eleven_turbo_v2_5",
            output_format="mp3_44100_128",
            voice_settings=VoiceSettings(
                stability=0.5,
                similarity_boost=0.75,
                style=0.0,
                speed=1.15,
                use_speaker_boost=True
            )
        )
        
        return response
    except Exception as e:
        raise Exception(f"Error generating audio: {str(e)}")


def save_audio_to_file(response, file_path: str) -> str:
    """
    Save audio response to a file.
    
    Parameters
    ----------
    response
        Audio response from ElevenLabs API
    file_path : str
        Path to save the audio file
        
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


def response_to_audio_bytes(response) -> bytes:
    """
    Convert API response to audio bytes.
    
    Parameters
    ----------
    response
        Audio response from ElevenLabs API
        
    Returns
    -------
    bytes
        Audio bytes ready to be used with st.audio
        
    Raises
    ------
    Exception
        If there's an error during conversion
    """
    try:
        # Handle both streaming and non-streaming responses
        if hasattr(response, '__iter__') and not isinstance(response, (bytes, str)):
            # For streaming response, collect chunks into a BytesIO object
            buffer = io.BytesIO()
            for chunk in response:
                if chunk:
                    buffer.write(chunk)
            buffer.seek(0)
            return buffer.read()
        else:
            # For complete bytes response
            return response
    except Exception as e:
        raise Exception(f"Error converting response to bytes: {str(e)}")


def text_to_audio_bytes(text: str, voice_id: str = "0S5oIfi8zOZixuSj8K6n") -> bytes:
    """
    Convert text to audio bytes using ElevenLabs API.
    
    Parameters
    ----------
    text : str
        The text to convert to speech
    voice_id : str, default="0S5oIfi8zOZixuSj8K6n"
        ElevenLabs voice ID to use for speech synthesis
        
    Returns
    -------
    bytes
        Audio bytes ready to be used with st.audio
        
    Raises
    ------
    Exception
        If there's an error during audio generation
    """
    response = get_audio_response(text, voice_id)
    return response_to_audio_bytes(response)


def text_to_audio_file(text: str, file_path: str, voice_id: str = "0S5oIfi8zOZixuSj8K6n") -> str:
    """
    Convert text to audio and save directly to a file.
    
    Parameters
    ----------
    text : str
        The text to convert to speech
    file_path : str
        Path to save the audio file
    voice_id : str, default="0S5oIfi8zOZixuSj8K6n"
        ElevenLabs voice ID to use for speech synthesis
        
    Returns
    -------
    str
        Path to the saved audio file
        
    Raises
    ------
    Exception
        If there's an error during audio generation or saving
    """
    response = get_audio_response(text, voice_id)
    return save_audio_to_file(response, file_path)


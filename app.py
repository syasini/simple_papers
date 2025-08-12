import numpy as np
import streamlit as st
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from loguru import logger
from streamlit_pdf_viewer import pdf_viewer
from simple_papers.paper import Paper
from simple_papers.annotation import Annotation
from simple_papers.summarizer import Summarizer
from simple_papers.audio_handler import AudioHandler
from simple_papers.utils import text_to_audio_bytes

def get_available_voices_for_section(group_id: str, audio_mapping: Dict[str, List[str]]) -> Tuple[List[str], Dict[str, str]]:
    """
    Extract available voices for a section from the audio mapping.
    
    Parameters
    ----------
    group_id : str
        The section ID to get voices for
    audio_mapping : Dict[str, List[str]]
        The audio mapping dictionary
        
    Returns
    -------
    Tuple[List[str], Dict[str, str]]
        A tuple containing a list of voice names and a dictionary mapping voice names to file paths
    """
    voice_options = []
    voice_paths = {}
    
    if group_id in audio_mapping and audio_mapping[group_id]:
        audio_files = audio_mapping[group_id]
        
        for audio_path in audio_files:
            # The voice is the last part before the .mp3 extension
            voice_name = Path(audio_path).stem.split('_')[-1]
            voice_options.append(voice_name)
            voice_paths[voice_name] = audio_path
    
    return voice_options, voice_paths

# set up wide mode
st.set_page_config(layout="wide")
st.sidebar.image("media/logo_small.png")
# st.sidebar.title("Simple papers")

# UI Configuration
with st.sidebar.expander("Display Settings"):
    container_height = st.number_input("Container Height", min_value=100, max_value=2000, value=950, step=50)
    zoom_level = st.number_input("Zoom Level", min_value=1.0, max_value=2.0, value=1.2, step=0.1)
    show_page_separator = st.checkbox("Show Page Separator", value=True)

# Initialize session state variables
if "is_parsed" not in st.session_state:
    st.session_state.is_parsed = False
    
if "summarizer" not in st.session_state:
    st.session_state.summarizer = None
    
if "annotations_list" not in st.session_state:
    st.session_state.annotations_list = []
    
if "summary_audio" not in st.session_state:
    st.session_state.summary_audio = None

if "selected_paper" not in st.session_state:
    st.session_state.selected_paper = None

if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None
    
# No need for audio_enabled in session state

# Load paper titles mapping from JSON file
try:
    with open("papers/paper_titles.json", "r") as f:
        paper_titles = json.load(f)
except Exception as e:
    logger.warning(f"Could not load paper titles: {e}")
    paper_titles = {}

st.sidebar.header("Paper Settings")


# File upload section
st.sidebar.markdown("#### Bring Your Own Paper (BYOP)")
uploaded_file = st.sidebar.file_uploader("Bring Your Own Paper (BYOP)", type="pdf", label_visibility="collapsed")

# Reset state when a file is uploaded
if uploaded_file is not None:
    # Always reset on upload to ensure clean state
    st.session_state.last_uploaded_file = uploaded_file.name
    st.session_state.is_parsed = False
    st.session_state.annotations_list = []
    st.session_state.summarizer = None
    st.session_state.summary_audio = None
    st.session_state.selected_paper = None  # Clear selection when uploading

# Paper selection dropdown (only shown if no file is uploaded)
# st.sidebar.markdown("### Paper arXiv", help="Yes it's a pun!")
selected_paper_id = None

if not uploaded_file and paper_titles:
    # Create options list from paper titles
    titles = [info["title"] for info in paper_titles.values()]
    
    # Show dropdown
    st.sidebar.caption(" or")
    st.sidebar.markdown("#### Select From the Paper arXiv", help="Yes it's a pun!")
    selected_title = st.sidebar.selectbox(
        "Select From the Paper arXiv",
        options=titles,
        label_visibility="collapsed",
        help="Yes it's a pun!",

    )
    
    # Find paper_id for selected title
    for paper_id, info in paper_titles.items():
        if info["title"] == selected_title:
            selected_paper_id = paper_id
            st.sidebar.caption(f"Paper ID: {selected_paper_id}")
            break
    
    
    # Update session state if selection changed
    if selected_paper_id != st.session_state.selected_paper:
        st.session_state.selected_paper = selected_paper_id
        # Reset state to ensure we load the correct annotations for the new paper
        st.session_state.is_parsed = False
        st.session_state.annotations_list = []
        st.session_state.summarizer = None
        st.session_state.summary_audio = None
        st.rerun()  # Force rerun to ensure clean slate for the new paper

# Create Paper instance based on selection or upload
if uploaded_file is not None:
    # paper_title = st.sidebar.text_input("Paper Title", value="")
    # Uploaded file takes precedence
    paper = Paper.from_file_uploader(uploaded_file)
elif selected_paper_id:
    # Use selected paper from dropdown
    paper_path = paper_titles[selected_paper_id]["path"]
    paper = Paper(paper_path)
else:
    # Default paper as fallback
    paper = Paper("papers/2210.03629/2210.03629.pdf")

# Check if this document has annotations or has been parsed
has_annotations = paper.has_annotations()

# st.sidebar.divider()

parse_doc_button = st.sidebar.empty()
parse_doc_info = st.sidebar.empty()

st.sidebar.divider()
# Add audio toggle to sidebar
st.sidebar.header("Paper Actions")
summarize_paper_button = st.sidebar.empty()

audio_enabled = st.sidebar.toggle("Enable Audio", value=False)
default_audio_voice = st.sidebar.selectbox("Default Audio Voice", 
                        options=["Alloy", "Joe", "Felicity", "Amelia", "Hope"], 
                        disabled=not audio_enabled)

if st.session_state.summary_audio:
    st.session_state.summary_audio.default_audio_voice = default_audio_voice
    

if has_annotations and not st.session_state.is_parsed:
    logger.info(f"Found existing annotations at {paper.path_handler.annotations_path}")
    
    # Initialize annotation handler and load annotations - this doesn't need parsed_doc
    annotation_handler = Annotation(paper)
    st.session_state.annotations_list = annotation_handler.annotations
    
    # Initialize summarizer with annotations
    st.session_state.summarizer = Summarizer(paper.path, st.session_state.annotations_list)
    
    # Initialize summary audio handler
    st.session_state.summary_audio = AudioHandler(paper.path, default_audio_voice=default_audio_voice)
    
    # Set state to parsed
    st.session_state.is_parsed = True
    
    logger.info("Paper is now ready with existing annotations")
    st.sidebar.success("✓ Paper loaded with existing annotations")

# Display PDF layout
col_l, col_r = st.columns(2)
col_l.write("## This scary looking paper 👇 ...")
# col_r.caption("[click on the annotations]")

# Get binary PDF for display (doesn't trigger parsing)
binary_pdf = paper.pdf_binary

# Parse Document button - only enabled if paper doesn't have annotations
if parse_doc_button.button("Parse Document", key="parse_document", disabled=paper.has_annotations()):
    with st.spinner("Parsing document..."):
        # This is where parsing happens
        paper.get_parsed_doc()  # This will parse the document and create the .pkl file if needed
        
        # Initialize annotation handler after parsing
        annotation_handler = Annotation(paper)
        # Generate annotations from the parsed document
        st.session_state.annotations_list = annotation_handler.generate_annotations()
        
        # Initialize summarizer with annotations
        st.session_state.summarizer = Summarizer(paper.path, st.session_state.annotations_list)
        
        # Initialize summary audio handler
        st.session_state.summary_audio = AudioHandler(paper.path, default_audio_voice=default_audio_voice)
        
        # Add paper to paper titles registry
        try:
            paper_id = Path(paper.path).stem.lower()
            paper_title = getattr(paper.parsed_doc, 'paper_title', f"Paper {paper_id}")
            
            # Load current registry
            try:
                with open("papers/paper_titles.json", "r") as f:
                    paper_titles = json.load(f)
            except Exception:
                paper_titles = {}
            
            # Add or update paper entry
            paper_titles[paper_id] = {
                "title": paper_title,
                "path": str(paper.path)
            }
            
            # Save updated registry
            with open("papers/paper_titles.json", "w") as f:
                json.dump(paper_titles, f, indent=2)
                
            logger.info(f"Added paper '{paper_title}' to registry")
        except Exception as e:
            logger.error(f"Error adding paper to registry: {str(e)}")
    
    st.rerun()


# Summarize Document button - only show if document is parsed
if st.session_state.is_parsed:
    # st.sidebar.markdown("### Document Actions")
    if summarize_paper_button.button("Summarize the Paper", key="summarize_document"):
        with st.spinner("Summarizing document..."):
            st.session_state.summarizer.summarize_all_sections()
        st.rerun()

    
    # Generate All Audio button - always show but disable when audio is off
    if st.sidebar.button("Generate All Audio", key="generate_all_audio", disabled=not audio_enabled):
        with st.spinner("Generating audio for all summaries..."):
            # Initialize the summary audio handler if not done yet
            if st.session_state.summary_audio is None:
                st.session_state.summary_audio = AudioHandler(paper_path=paper.path, default_audio_voice=default_audio_voice)
                st.write(st.session_state.summary_audio.voice_mapping)
            # Get all summaries from the summarizer
            summaries = st.session_state.summarizer._summaries
            
            if summaries:
                # Generate audio for all summaries
                audio_paths = st.session_state.summary_audio.generate_all_audio(summaries)
                st.sidebar.success(f"✓ Generated audio for {len(audio_paths)} summaries")
            else:
                st.sidebar.warning("No summaries found. Generate summaries first.")
                
    if not audio_enabled:
        st.sidebar.info("Enable audio to generate and play audio summaries.")


def show_annotation(annotation):
    """Handle annotation click and display summary and audio"""
    if not st.session_state.is_parsed or not st.session_state.summarizer:
        return
    
    group_id = annotation["group"]
    summary = st.session_state.summarizer.summarize_section(group_id)
    col_r.write("## is actually pretty simple! 💁")

    with col_r.container(height=container_height):
        summary_tab, markdown_tab, raw_json_tab = st.tabs(["Summary", "Markdown", "Raw JSON"])

        summary_tab.markdown(summary)
        markdown_tab.markdown(annotation["group_text"])
        raw_json_tab.json(annotation)



        if summary_tab.button("Regenerate Summary", key=f"regenerate_summary_{group_id}", type="primary"):
            summary = st.session_state.summarizer.summarize_section(group_id, override_summary=True)
            if st.session_state.summary_audio:
                st.session_state.summary_audio.delete_audio_file(group_id)
            st.rerun()
        
        # Use the default voice selected in the sidebar
        selected_voice = default_audio_voice
        
        # Regenerate audio button with selected voice
        regenerate_audio_button = summary_tab.button(f"[Re]generate Audio ({selected_voice})", key=f"regenerate_audio_{group_id}", 
                type="primary", disabled=not audio_enabled, 
                help=f"This will regenerate the audio for the selected section using the '{selected_voice}' voice")
        if regenerate_audio_button:
            if st.session_state.summary_audio:
                with st.spinner("Generating audio..."):
                    st.session_state.summary_audio.delete_audio_file(group_id, voice=selected_voice)
                    # Set the voice as the default temporarily for this generation
                    original_voice = st.session_state.summary_audio.default_audio_voice
                    st.session_state.summary_audio.default_audio_voice = selected_voice
                    st.session_state.summary_audio.save_summary_audio_to_file(group_id, summary)
                    # Restore original default voice
                    st.session_state.summary_audio.default_audio_voice = original_voice

            st.rerun()
        
        # Only show audio if enabled and summary_audio handler is initialized
        if audio_enabled and st.session_state.summary_audio:
            # Check if we have audio files for this section
            audio_mapping = st.session_state.summary_audio._audio_mapping
            if group_id in audio_mapping and audio_mapping[group_id]:
                # Get available voices for this section using utility function
                voice_options, voice_paths = get_available_voices_for_section(group_id, audio_mapping)
                
                # Show dropdown to select voice if multiple options available
                if len(voice_options) > 1:
                    selected_audio_voice = summary_tab.selectbox(
                        "Available Audio Narrations",
                        options=voice_options,
                        key=f"audio_select_{group_id}"
                    )
                    # Get the selected audio file path
                    selected_path = voice_paths.get(selected_audio_voice)
                    if selected_path:
                        with open(selected_path, "rb") as f:
                            audio_bytes = f.read()
                        summary_tab.audio(audio_bytes, format="audio/mp3")
                else:
                    # If only one audio file, just play it directly
                    session_audio = st.session_state.summary_audio.get_audio_bytes(group_id)
                    summary_tab.audio(session_audio, format="audio/mp3")
            else:
                summary_tab.info("No audio available for this section. Generate audio first.")
        elif not audio_enabled and st.session_state.summary_audio:
            summary_tab.info("Audio is disabled. Toggle 'Enable Audio' in the sidebar to hear summaries.")
        elif not audio_enabled and st.session_state.summary_audio:
            summary_tab.info("Audio is disabled. Toggle 'Enable Audio' in the sidebar to hear summaries.")



with col_l.container(height=container_height):
    pdf_viewer(
        binary_pdf,
        width=700,
        height=1000,
        # Only use annotations if document is parsed
        annotations=st.session_state.annotations_list if st.session_state.is_parsed else [], 
        annotation_outline_size=2.5,
        zoom_level=zoom_level,
        show_page_separator=show_page_separator,
        on_annotation_click=show_annotation,
    )

# Add an appropriate message based on the document state
if not st.session_state.is_parsed:
    if paper.has_annotations():
        parse_doc_info.info("⚠️ Annotations exist but haven't been loaded yet. Please reload the page.")
    else:
        parse_doc_info.info("📋 Click 'Parse Document' to analyze the paper and see annotations.")


# st.write(st.session_state["summarizer"]._summaries)


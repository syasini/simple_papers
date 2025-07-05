import numpy as np
import streamlit as st
import os
import json
from pathlib import Path
from loguru import logger
from streamlit_pdf_viewer import pdf_viewer
from simple_papers.paper import Paper
from simple_papers.annotation import Annotation
from simple_papers.summarizer import Summarizer
from simple_papers.audio_handler import AudioHandler
from simple_papers.utils import text_to_audio_bytes

# set up wide mode
st.set_page_config(layout="wide")
st.sidebar.image("media/logo_small.png")
# st.sidebar.title("Simple papers")

# UI Configuration
container_height = st.sidebar.number_input("Container Height", min_value=100, max_value=2000, value=800, step=50)
zoom_level = st.sidebar.number_input("Zoom Level", min_value=1.0, max_value=2.0, value=1.2)
show_page_separator = st.sidebar.checkbox("Show Page Separator", value=True)

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

# File upload section
st.sidebar.markdown("### Upload PDF")
uploaded_file = st.sidebar.file_uploader("Upload a new paper", type="pdf")

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
st.sidebar.markdown("### Paper arXiv")
selected_paper_id = None

if not uploaded_file and paper_titles:
    # Create options list from paper titles
    titles = [info["title"] for info in paper_titles.values()]
    
    # Show dropdown
    selected_title = st.sidebar.selectbox(
        "Select a paper",
        options=titles
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

parse_doc_button = st.sidebar.empty()


# Add audio toggle to sidebar
st.sidebar.markdown("### Audio Settings")
audio_enabled = st.sidebar.toggle("Enable Audio", value=False)
default_audio_voice = st.sidebar.selectbox("Default Audio Voice", options=["Alloy", "Joe"], disabled=not audio_enabled)

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
    st.sidebar.markdown("### Document Actions")
    if st.sidebar.button("Summarize Document", key="summarize_document"):
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
        

        # Only show audio if enabled and summary_audio handler is initialized
        if audio_enabled and st.session_state.summary_audio:
            session_audio = st.session_state.summary_audio.get_audio_bytes(group_id, summary)
            summary_tab.audio(session_audio, format="audio/mp3")
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
        st.sidebar.info("⚠️ Annotations exist but haven't been loaded yet. Please reload the page.")
    else:
        st.sidebar.info("📋 Click 'Parse Document' to analyze the paper and see annotations.")


# st.write(st.session_state["summarizer"]._summaries)


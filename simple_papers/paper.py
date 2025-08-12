import os
import pickle
from pathlib import Path
from loguru import logger
from typing import List, Dict, Any, Optional
from functools import cached_property
from agentic_doc.common import ParsedDocument

# Import local modules
from simple_papers.path_handler import PathHandler
from simple_papers.annotation import Annotation
from simple_papers.summarizer import Summarizer
from simple_papers.audio_handler import AudioHandler
from simple_papers.utils import load_pdf_to_binary, get_page_dimensions_pymupdf


class Paper:
    """Main class to handle papers, their documents, annotations, summaries, and audio.
    
    This class serves as the central coordinator for all paper-related functionality,
    delegating specialized tasks to the appropriate modules.
    """
    
    def __init__(self, path: str, parsed_doc: ParsedDocument = None):
        """
        Initialize a Paper object for handling PDF documents.
        
        Parameters
        ----------
        path : str
            Path to the PDF file
        parsed_doc : ParsedDocument, optional
            Parsed document if already available
        """
        self.path = path
        self.path_handler = PathHandler(path)
        
        self.parsed_doc_path = self.path_handler.parsed_doc_path
        # Don't load parsed_doc on initialization unless explicitly provided
        self.parsed_doc = parsed_doc if parsed_doc is not None else None

        self.pdf_binary = load_pdf_to_binary(path)

    def load_parsed_doc_if_exists(self) -> Optional[ParsedDocument]:
        """
        Load the parsed document from disk if it exists.
        This doesn't parse the document, just loads an existing .pkl file.
        
        Returns
        -------
        ParsedDocument or None
            The parsed document if available on disk, None otherwise
        """
        if self.parsed_doc_path.exists():
            logger.info(f"Loading parsed document from {self.parsed_doc_path}")
            try:
                with open(self.parsed_doc_path, "rb") as f:
                    self.parsed_doc = pickle.load(f)
                return self.parsed_doc
            except Exception as e:
                logger.error(f"Error loading parsed document: {str(e)}")
                return None
        else:
            logger.info(f"No parsed document found at {self.parsed_doc_path}")
            return None

    def fetch_parsed_doc(self, parsed_doc: ParsedDocument = None):
        """
        Fetch the parsed document only if needed.
        
        Parameters
        ----------
        parsed_doc : ParsedDocument, optional
            Parsed document if already available
            
        Returns
        -------
        ParsedDocument or None
            The parsed document if available
        """
        if parsed_doc is not None:
            logger.info("Parsed document provided")
            return parsed_doc
        elif self.parsed_doc is not None:
            logger.info("Parsed document already loaded")
            return self.parsed_doc
        else:
            return self.load_parsed_doc_if_exists()

    @cached_property
    def paper_width(self):
        """Get the width of the first page of the PDF."""
        width, _ = get_page_dimensions_pymupdf(self.path)
        return width
        
    @cached_property
    def paper_height(self):
        """Get the height of the first page of the PDF."""
        _, height = get_page_dimensions_pymupdf(self.path)
        return height

    def get_parsed_doc(self):
        """
        Get or create the parsed document.
        If document is not already loaded, try to load from disk.
        If it doesn't exist on disk, parse it and save it.
        
        Returns
        -------
        ParsedDocument
            The parsed document
        """
        # First try to get from memory
        if self.parsed_doc is not None:
            return self.parsed_doc
            
        # Then try to load from disk
        if self.parsed_doc_path.exists():
            return self.load_parsed_doc_if_exists()
            
        # If all else fails, parse the document
        logger.info("No parsed document found, parsing from scratch")
        self.parsed_doc = self._parse_document(self.path)
        return self.parsed_doc

    def get_parse_document(self):
        """For backwards compatibility"""
        return self.get_parsed_doc()
        
    def has_annotations(self) -> bool:
        """
        Check if annotations exist for this paper.
        
        Returns
        -------
        bool
            True if annotations file exists, False otherwise
        """
        return self.path_handler.annotations_path.exists()

    @staticmethod
    def _parse_document(path: str) -> ParsedDocument:
        """
        Parse the document using agentic_doc.parse.parse
        
        Parameters
        ----------
        path : str
            The path to the document to parse
        
        Returns
        -------
        ParsedDocument
            The parsed document
        """
        logger.info(f"Parsing document: {path}")
        try:
            from agentic_doc.parse import parse
            
            # Parse the document
            parsed_document = parse(path)
            
            # Cache the parsed document to disk
            path_handler = PathHandler(path)
            parsed_doc_path = path_handler.parsed_doc_path
            with open(parsed_doc_path, "wb") as f:
                pickle.dump(parsed_document, f)
            
            logger.info(f"Parsed document saved to {parsed_doc_path}")
            return parsed_document
            
        except Exception as e:
            logger.error(f"Error parsing document: {str(e)}")
            raise
    
    @classmethod
    def from_file_uploader(cls, uploaded_file, parsed_doc: ParsedDocument = None):
        """
        Create a Paper object from a file uploaded through streamlit's file_uploader.
        
        Parameters
        ----------
        uploaded_file : UploadedFile
            The uploaded file object from streamlit's file_uploader
        parsed_doc : ParsedDocument, optional
            The parsed document, by default None
            
        Returns
        -------
        Paper
            The Paper object with the relative path for better portability
        """
        from simple_papers.path_handler import PathHandler
        
        # Get original filename
        original_filename = uploaded_file.name
        
        # Create paths for the uploaded file using PathHandler
        # Get both absolute path (for writing) and relative path (for storage)
        abs_file_path, rel_file_path, _ = PathHandler.create_path_for_uploaded_file(original_filename)
        
        # Write the uploaded file to disk using the absolute path
        with open(abs_file_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # Reset the file pointer for future reads if needed
        uploaded_file.seek(0)
        
        logger.info(f"Saved uploaded file to {abs_file_path}")
        logger.info(f"Using relative path {rel_file_path} for paper registry")
        
        # Create a Paper object from the relative path for better portability
        return cls(rel_file_path, parsed_doc)
    
    def get_annotation(self) -> Annotation:
        """
        Get an Annotation object for this paper.
        
        Returns
        -------
        Annotation
            The Annotation object for this paper
        """
        return Annotation(self)
    
    def get_summarizer(self, annotations_list: List[Dict[str, Any]]) -> Summarizer:
        """
        Get a Summarizer object for this paper.
        
        Parameters
        ----------
        annotations_list : List[Dict[str, Any]]
            List of annotation dictionaries from Annotation.annotations
            
        Returns
        -------
        Summarizer
            The Summarizer object for this paper
        """
        return Summarizer(self.path, annotations_list)
    
    def get_audio_handler(self) -> AudioHandler:
        """
        Get an AudioHandler object for this paper.
        
        Returns
        -------
        AudioHandler
            The AudioHandler object for this paper
        """
        return AudioHandler(self.path)

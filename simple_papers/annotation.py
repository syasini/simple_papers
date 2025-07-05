import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger
from collections import defaultdict

from agentic_doc.common import ParsedDocument

from simple_papers.path_handler import PathHandler
from simple_papers.utils import get_annotation_dimensions


class Annotation:
    """Class to handle annotations from parsed documents.
    It processes the parsed document to create annotations and handle merging/summarizing sections."""
    
    def __init__(self, paper):
        """
        Initialize an Annotation object.
        
        Parameters
        ----------
        paper : Paper
            The paper object containing the PDF and parsed document
        """
        self.paper = paper
        self.annotations_path = self.paper.path_handler.annotations_path

        self._annotations_cache: Optional[List[Dict[str, Any]]] = None
        self._group_cache: Optional[Dict[str, int]] = None
        self._group_texts: Optional[Dict[int, List[str]]] = None
        
        # Load annotations if they exist
        if self.annotations_path.exists():
            self._load_annotations_from_file()
        
    @property
    def annotations(self) -> List[Dict[str, Any]]:
        """
        Get annotations from memory cache, file, or generate them if needed.
        
        Returns
        -------
        list
            List of annotation dictionaries compatible with the PDF viewer
        """
        if self._annotations_cache:
            logger.info("Annotations already exist in memory cache")
            return self._annotations_cache
        elif self.annotations_path.exists():
            logger.info(f"Annotations already exist in file {self.annotations_path}")
            # Load annotations from file
            return self._load_annotations_from_file()
        else:
            # Generate annotations if they don't exist
            logger.info("No annotations found, generating them")
            return self.generate_annotations()
        
    def generate_annotations(self) -> List[Dict[str, Any]]:
        """
        Generate annotations from parsed document.
        Process the parsed document to create annotations for PDF viewer.
        
        Returns
        -------
        list
            List of annotation dictionaries compatible with the PDF viewer
        """
        logger.info("Generating new annotations")
        
        # This requires the parsed document
        parsed_doc = self.paper.get_parsed_doc()
        if parsed_doc is None:
            logger.error("Cannot generate annotations: parsed document is None")
            return []
        
        # First pass: Create initial annotations with group ids
        annotations = []
        chunk_to_group = self.get_chunk_group_mapping(parsed_doc)
        
        # Create a defaultdict to collect texts by group
        group_texts = defaultdict(list)
        
        for chunk in parsed_doc[0].chunks:
            if chunk.chunk_type.name != "text":
                continue
                
            chunk_groundings = chunk.grounding
            chunk_id = chunk.chunk_id
            text = chunk.text
            group = chunk_to_group[chunk_id]
            
            # Add this text to the group's text list (no need to initialize with defaultdict)
            group_texts[group].append(text)
            
            for grounding in chunk_groundings:
                # Get dimensions using the utility function
                dims = get_annotation_dimensions(
                    grounding.box.l, grounding.box.t, grounding.box.r, grounding.box.b, 
                    self.paper.paper_width, self.paper.paper_height
                )
                
                # Extract the number part for color alternation
                group_num = int(group.split('-')[0])
                # switch color between yellowgreen and orange
                color = "yellowgreen" if group_num % 2 == 0 else "orange"

                annotations.append({
                    "text": text,
                    "page": grounding.page+1,
                    **dims,  # Unpack x, y, width, height
                    "color": color,
                    "border": "dashed",
                    "background_color": "white",
                    "group": group,
                    "chunk_id": chunk_id,
                    # We'll add group_text in the second pass
                })
        
        # Second pass: Add group_text to each annotation
        for annotation in annotations:
            group = annotation["group"]
            # Always include the group_text field
            annotation["group_text"] = "\n\n".join(group_texts[group])

        logger.info(f"Generated {len(annotations)} annotations")
        
        self._annotations_cache = annotations

        # Save annotations to file for future use
        with open(self.annotations_path, "w") as f:
            json.dump({"annotations": annotations}, f, indent=4)
        return annotations

    
    def _load_annotations_from_file(self) -> List[Dict[str, Any]]:
        """
        Load annotations from the JSON file.
        
        Returns
        -------
        list
            List of annotation dictionaries
        """
        try:
            with open(self.annotations_path, "r") as f:
                annotations_data = json.load(f)
                self._annotations_cache = annotations_data["annotations"]
                return self._annotations_cache
        except Exception as e:
            logger.error(f"Error loading annotations from {self.annotations_path}: {str(e)}")
            return []
    
    def get_chunk_group_mapping(self, parsed_doc: ParsedDocument = None) -> Dict[str, str]:
        """
        Get the chunk groups, using cached results if available.
        
        Parameters
        ----------
        parsed_doc : ParsedDocument, optional
            The parsed document to group
            
        Returns
        -------
        Dict[str, str]
            Dictionary mapping chunk IDs to group identifiers
        """
        # If we have a cache, return it
        if self._group_cache is not None:
            return self._group_cache
        
        # If no parsed_doc provided but needed, try to get it
        if parsed_doc is None:
            parsed_doc = self.paper.get_parsed_doc()
            if parsed_doc is None:
                logger.error("Cannot get chunk group mapping: no parsed document available")
                return {}
            
        # Generate the grouping and cache it
        self._group_cache = self._group_parsed_doc_chunks(parsed_doc)
        return self._group_cache
    
    def _group_parsed_doc_chunks(self, parsed_doc: ParsedDocument) -> Dict[str, str]:
        """
        Group document chunks into logical sections by identifying section headers.
        
        We increment the group number when we encounter:
        1. Abstract section
        2. Numbered sections (e.g., "1. Introduction", "2.3 Methods")
        3. Conclusion, References, or Index sections
        
        Returns
        -------
        Dict[str, str]
            Dictionary mapping chunk IDs to group identifiers (e.g., "0-title", "1-abstract")
        """
        logger.debug("Grouping parsed document chunks")
        if parsed_doc is None or len(parsed_doc) == 0:
            return {}
            
        # Initialize chunk ID to group identifier mapping
        chunk_to_group = {}
        chunks = parsed_doc[0].chunks
        
        logger.info(f"Processing {len(chunks)} chunks for grouping")
        
        # Define regex pattern for numbered sections (e.g., "1. Introduction", "2.3 Methods")
        section_pattern = re.compile(r'^(#\s*)?\d+(\.\d+)?\.?\s+\w+', re.MULTILINE)
        
        # Define patterns for special sections
        special_sections = ["abstract", "conclusion", "discussion", "reference", "bibliography", "index", "appendix"]
        
        # Initialize group counter and current section name
        current_group = 0
        current_section = "title"  # Default to title for group 0
        
        # Helper function to check if text is a section header and identify section type
        def get_section_type(text: str) -> str:
            # Remove any leading # for markdown headers and clean text
            clean_text = re.sub(r'^#+\s*', '', text.strip().lower())
            
            # Check if it's one of the special sections
            for section in special_sections:
                if clean_text.startswith(section):
                    return section
            
            # For numbered sections, extract the title after the number
            if section_pattern.match(clean_text):
                # Try to extract a section name from the text
                # Example: from "1. Introduction" extract "introduction"
                match = re.search(r'^\d+(?:\.\d+)?\s*\.?\s*(\w+)', clean_text)
                if match:
                    return match.group(1).lower()
                return "section"  # Generic name if we can't extract
                
            return None  # Not a section header
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            # Skip non-text chunks
            if chunk.chunk_type.name != "text":
                continue
                
            # Get chunk ID or use index if not available
            chunk_id = getattr(chunk, 'chunk_id', str(i))
            text = chunk.text.strip()
            
            # Check if this chunk is a section header
            section_type = get_section_type(text)
            if section_type:
                # Increment group number for new section
                current_group += 1
                current_section = section_type
                
            # Create group identifier as "number-name"
            group_id = f"{current_group}-{current_section}"
                
            # Assign current group ID to this chunk
            chunk_to_group[chunk_id] = group_id
            
        return chunk_to_group
    
    def summarize_section(self, section_name: str) -> str:
        """
        Summarize a specific section of the document.
        
        Parameters
        ----------
        section_name : str
            Name of the section to summarize
            
        Returns
        -------
        str
            Summarized text content for the specified section
        """
        # This is a placeholder implementation 
        # You would need to implement the logic to identify the section and summarize it
        return f"Summary of {section_name} section"

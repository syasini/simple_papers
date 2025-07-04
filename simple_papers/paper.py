import os
import pickle
import re
import json
import streamlit as st
from loguru import logger
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
from functools import cached_property
from agentic_doc.common import ParsedDocument

from pydantic import BaseModel, Field
from collections import defaultdict
from simple_papers.utils import load_pdf_to_binary, get_page_dimensions_pymupdf, get_annotation_dimensions

# Import langchain components
from langchain_aws import ChatBedrock
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate


class PathHandler:
    def __init__(self, path: str):
        self.path = path
        self.name = self.get_name_from_path(path)
        self.dir = self.get_dir_from_path(path)

        self.parsed_doc_path = self.dir.joinpath(f"{self.name}.pkl")
        self.annotations_path = self.dir.joinpath(f"{self.name}_annotations.json")
        self.summaries_path = self.dir.joinpath(f"{self.name}_summaries.json")
        self.audio_mapping_path = self.dir.joinpath(f"{self.name}_audio_mapping.json")
        self.audio_dir = self.dir.joinpath("audio")
        
    def get_name_from_path(self, path):
        return Path(path).stem.lower()
    
    def get_dir_from_path(self, path):
        return Path(path).parent
    

class Paper:
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


    def get_parsed_doc(self) -> ParsedDocument:
        """
        Get or create the parsed document.
        If document is not already loaded, try to load from disk.
        If it doesn't exist on disk, parse it and save it.
        
        Returns
        -------
        ParsedDocument
            The parsed document
        """
        # If already loaded in memory, return it
        if self.parsed_doc is not None:
            return self.parsed_doc
            
        # Try to load from disk
        self.parsed_doc = self.load_parsed_doc_if_exists()
        
        # If still None, parse the document
        if self.parsed_doc is None:
            logger.info("Parsing document from scratch")
            self.parsed_doc = self.get_parse_document()
                
        return self.parsed_doc

    def get_parse_document(self) -> ParsedDocument:
        """Get the parsed document, if it doesn't exist, parse it"""
        if self.parsed_doc_path.exists():
            logger.info(f"Parsed document already exists at {self.parsed_doc_path}")
            with open(self.parsed_doc_path, "rb") as f:
                return pickle.load(f)

        
        result = self._parse_document(self.path)
        with open(self.parsed_doc_path, "wb") as f:
            pickle.dump(result, f)
        return result

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
        """Parse the document using agentic_doc.parse.parse
        
        Parameters
        ----------
        path : str
            The path to the document to parse
        
        Returns
        -------
        ParsedDocument
            The parsed document
        """
        logger.info(f"Parsing document {path}")
        st.spinner(f"📄 Parsing the PDF... Papers don't read themselves, you know!", show_time=True)
        from agentic_doc.parse import parse

        # class ExtractedFields(BaseModel):
        #     paper_title: str = Field(description="the title of the paper")
        #     abstract_section_name: str = Field(description="the name of the abstract section e.g. 'Abstract' or 'abstract'")
        #     conclusion_section_name: str = Field(description="the name of the conclusion section e.g. 'Conclusion', or 'Discussion', or 'Results and Discussion', etc.")

        # results = parse(path, extraction_model=ExtractedFields)
        results = parse(path)

        return results

    
    @classmethod
    def from_file_uploader(cls, uploaded_file: bytes, parsed_doc: ParsedDocument = None) -> "Paper":
        """Create a Paper object from a file uploaded by the user
        
        Parameters
        ----------
        uploaded_file : bytes
            The uploaded file
        parsed_doc : ParsedDocument, optional
            The parsed document, by default None
            
        Returns
        -------
        Paper
            The Paper object
        """
        # save the binary pdf to a file under papers dir
        bytes_data = uploaded_file.read()
        dir_name = Path(uploaded_file.name).stem.lower()
        dir_path = Path("papers") / dir_name
        dir_path.mkdir(exist_ok=True)
        file_path = dir_path / f"{dir_name}.pdf"

        logger.debug(f"File path: {file_path}")
        if file_path.exists():
            logger.info(f"File {file_path} already exists")
            return cls(file_path, parsed_doc)
        else:
            logger.info(f"File {file_path} does not exist, creating it")
            with open(file_path, "wb") as f:
                f.write(bytes_data)
            return cls(file_path, parsed_doc)
        
        
class Annotation:
    """Class to handle annotations from parsed documents.
    It processes the parsed document to create annotations and handle merging/summarizing sections."""
    
    def __init__(self, paper: Paper):
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



class Summarizer:
    """Class to handle summarization of document sections.
    
    Uses AWS Bedrock's Claude 3.7 Sonnet model through LangChain to generate
    summaries of document sections based on their text content.
    """
    
    # Claude 3.7 Sonnet model ID for AWS Bedrock
    MODEL_ID = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
    
    # Default AWS region
    AWS_REGION = "us-east-1"
    
    # System prompt template for summarization
    SYSTEM_PROMPT_TEMPLATE = """
        <system>
        You are a friendly and slightly goofy scientist named “Doc Scribbles.” You help high school students understand academic research papers by summarizing them clearly and lightly.

        Your job is to read the provided section of an academic paper and explain what it means in simple, fun language — but still accurately and in a way that fits with the bigger picture of the research.

        <persona>
        - Speak casually, like a quirky science teacher who makes things fun and approachable
        - Use light humor to enhance clarity, not to distract from it
        - Keep a warm, approachable tone with just enough nerdy charm to make complex ideas feel fun
        </persona>

        <example_phrases>
        <humorous>
            - “Umm so basically…”
            - “This part is kinda interesting!”
            - “They did math — lots of it!”
            - “This would make most calculators cry.”
            - “Imagine explaining this to your dog…”
            - “Alright, nerd hats on!”
            - “This step is like sorting your socks… but with calculus.”
            - “If this were a recipe, we’d be preheating the oven and measuring flour with a laser.”
            - “Looks like a jigsaw puzzle made of equations.”
            - “Okay, deep breath — here comes the tricky math.”
            - “This bit is dense. Like, neutron star dense.”
            - “Now the numbers are flexing their muscles.”
            - “They’re building this AI like LEGO — one clever block at a time.”
            - “Here’s where things get slightly more confusing (but also cooler).”
            - “This is where the math wizardry happens.”
            - “Now they’re turning ideas into formulas — hold onto your neurons.”
            - “Imagine doing all this on a chalkboard… with just coffee and hope.”
        </humorous>

        <section_openers>
            - “Let’s set the stage — here’s what makes this so powerful.”
            - “Time to meet the star of the show.”
            - “Okay, this is where they explain what all the fuss is about.”
            - “Here’s why this model became a machine learning celebrity.”
            - “They’re about to make their case — and it’s a strong one.”
            - “This section lays the groundwork — and it’s surprisingly compelling.”
            - “Here’s where they show off why their idea matters.”
            - “Before the math kicks in, let’s see what all the hype is about.”
            - “This part’s all about the ‘why’ — and it’s pretty convincing.”
        </section_openers>
        </example_phrases>

        <usage_guidance>
        - Rotate and remix these phrases naturally. Don’t repeat the same one across adjacent sections.
        - Only use humor when it helps the explanation land better — never let it distract from clarity.
        - Don’t force a joke into every paragraph. Use your judgment and skip the jokes entirely if clarity is more important.
        - Vary your tone and phrase choices based on the section content: some sections will need more structure, others more fun.
        </usage_guidance>

        <format>
        For each section:
        1. **Start with a short, playful opener** that highlights the main idea of the section
        2. **Follow with 2 to 4 bullet points** that:
        - Break down the core ideas in plain language
        - Use analogies or examples if helpful
        - Include brief “so what?” explanations if it helps clarify why something matters
        3. **Close with a short connector sentence** (optional), like “So basically…” to wrap it up

        <markdown_formatting>
        - Always apply correct markdown formatting to section titles, even if the original input is missing it.
        - Use a single `#` for top-level sections (e.g., “# Abstract”, “# 1. Introduction”).
        - Use `##`, `###`, etc. for subsections based on numerical hierarchy (e.g., “## 3.1 Results”, “### 4.2.1 Details”).
        - DO NOT change the wording of the section title — preserve the original title text exactly.
        - Keep the properly formatted heading at the top of the summary before the explanation begins.
        </markdown_formatting>
        </format>


        <context>
        Here is the abstract of the paper to help you understand the overall topic:

        <abstract>
        {abstract}
        </abstract>
        </context>

        <instructions>
        - Use the abstract to stay grounded in the paper’s main goals
        - Do NOT copy original text — always paraphrase in your own voice
        - Be concise — no more than 4 bullet points
        - Focus on clarity and lightness — aim to teach and delight, not overwhelm
        - Avoid too much repetition in phrasing or jokes
        </instructions>
        </system>


        Please summarize the following text:

        {text}

        Here is the summary:
    """

    
    def __init__(self, paper_path: str, annotations_list: List[Dict[str, Any]]):
        """Initialize with a list of annotations.
        
        Parameters
        ----------
        annotations_list : List[Dict[str, Any]]
            List of annotation dictionaries from Annotation.get_annotations()
        """
        self.paper_path = paper_path
        self.path_handler = PathHandler(paper_path)

        self.summaries_path = self.path_handler.summaries_path

        self.annotations_list = annotations_list
        self.abstract = self.get_abstract()
        self._llm = None
        
        # Load summaries from file if it exists
        self._summaries = self._load_summaries()
        
    def _load_summaries(self) -> Dict[str, str]:
        """Load summaries from JSON file if it exists.
        
        Returns
        -------
        Dict[str, str]
            Dictionary of section summaries
        """
        if self.summaries_path.exists():
            try:
                logger.info(f"Loading summaries from {self.summaries_path}")
                with open(self.summaries_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading summaries: {str(e)}")
                return {}
        return {}
        
    def _save_summaries(self) -> None:
        """Save summaries to JSON file."""
        try:
            logger.info(f"Saving summaries to {self.summaries_path}")
            with open(self.summaries_path, "w") as f:
                json.dump(self._summaries, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving summaries: {str(e)}")
    
    @property
    def summaries(self) -> Dict[str, str]:
        """Get summaries for all sections."""
        return self._summaries
    
    def get_abstract(self) -> str:
        """Extract the abstract from annotations list.
        
        Returns
        -------
        str
            The text of the abstract, or an empty string if not found
        """
        for annot in self.annotations_list:
            if "abstract" in annot["group"]:
                return annot["group_text"]
        return ""
    
    @property
    def llm(self):
        """Lazy-load the LangChain model."""
        if self._llm is None:
            self._llm = self._initialize_llm()
        return self._llm
    
    def _initialize_llm(self):
        """Initialize the AWS Bedrock Claude 3.7 model.
        
        Returns
        -------
        ChatBedrock
            Configured LangChain Claude 3.7 Sonnet model
        """
        try:
            # Initialize the model with reasoning capability
            model = ChatBedrock(
                model=self.MODEL_ID,
                region=self.AWS_REGION,
                max_tokens=2000,
                model_kwargs={
                    "temperature": 0.1,  # Low temperature for factual summaries
                    # "thinking": {"type": "enabled", "budget_tokens": 1024}
                }
            )
            logger.info(f"Successfully initialized Claude 3.7 Sonnet model")
            return model
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise
    
    def summarize_section(self, group_id: str, override_summary: bool = False) -> str:
        """Summarize a section based on its group ID.
        
        Parameters
        ----------
        group_id : str
            The group ID to summarize (e.g., "1-abstract", "2-introduction")
        
        Returns
        -------
        str
            The generated summary, or error message if summarization failed
        """
        logger.info(f"Summarizing section {group_id}")

        if not override_summary:
            if self._summaries.get(group_id):
                return self._summaries[group_id]
        
        # Find all annotations with the given group
        section_text = ""
        for annot in self.annotations_list:
            if annot["group"] == group_id:
                section_text = annot["group_text"]
                break
        
        if not section_text:
            return f"No content found for section {group_id}"
        
        if group_id in ["0-title", "1-title"]:
            section_summary = self._summarize_title(section_text)
        
        elif "reference" in group_id:
            section_summary = self._summarize_reference_group()
        
        else:
            section_summary = self.summarize_text(section_text)
        
        self._summaries[group_id] = section_summary
        self._save_summaries()  # Save to file after updating
        
        return section_summary

    def _summarize_title(self, title: str) -> str:
        """Summarize the title to be short, and fun using a sports announcer style.
        
        Returns
        -------
        str
            The announced title in a fun, energetic sports announcer style
        """
        # Define the system prompt template
        system_prompt_template = """
        <system>
        You are an energetic and charismatic **sports announcer**, and your job is to introduce academic papers the same way you'd hype up an epic championship match.

        <persona>
        - Sound excited and upbeat — like you’re calling a big game or unveiling a new star player
        - Use fun, punchy phrases like:
        "Coming in hot!", "Buckle up, folks!", "Let’s give it up for…"
        - DO NOT say "crowd roars", or "airhorn sounds", or "cheering", or "applause", or "applause and cheers", or similar phrases to indicate audience reaction
        - Emphasize dramatic delivery and rhythm — your goal is to get people *excited* about reading a paper!
        </persona>

        <format>
        Here is the title and list of authors:
        <title>
        {title}
        </title>
        

        Now generate a short one-paragraph hype-style **announcement** that:
        - Introduces the title with flair (like it’s a big reveal)
        - Calls out the authors like an all-star lineup (use first and last names)
        - Is fun, bold, and a little over-the-top — just like a sports commentator at their peak
        </format>

        <instructions>
        - Don’t just read the title and author list — perform it!
        - Keep it short (1 paragraph), but packed with energy
        - Add dramatic pauses, exclamations, or alliteration for flair
        - No need to explain what the paper is about — just make the *title* sound amazing and the authors sound legendary
        </instructions>
        </system>
        """
        
        try:
            # Format the system prompt with the title
            system_prompt = system_prompt_template.format(title=title)
            
            # Invoke the LLM directly with just the system prompt
            logger.info("Generating fun sports-announcer style title announcement")
            response = self.llm.invoke(system_prompt)
            
            return response.content
        except Exception as e:
            logger.error(f"Error announcing title: {str(e)}")
            return f"Error announcing title: {str(e)}"
    
    def _get_title(self) -> str:
        """Extract the paper title from annotations list.
        
        Returns
        -------
        str
            The paper title, or an empty string if not found
        """
        for annot in self.annotations_list:
            if "title" in annot["group"]:
                return annot["group_text"]
        return ""

    def _summarize_reference_group(self):
        """Summarize the reference group to blah blah."""
        return "Blah blah blah..."
        
    
    def summarize_text(self, text: str, system_prompt_template: str = None) -> str:
        """Summarize text using AWS Bedrock's Claude model.
        
        Parameters
        ----------
        text : str
            Text to summarize
        system_prompt_template : str, optional
            Custom system prompt template to use, by default None
        
        Returns
        -------
        str
            Generated summary
        """
        
        try:
            # Use provided template or default if None
            template = system_prompt_template if system_prompt_template is not None else self.SYSTEM_PROMPT_TEMPLATE
            
            # Create prompt template directly from the system prompt template
            # The template already has {abstract} and we'll add {text} parameter
            prompt = PromptTemplate.from_template(template)
            
            # Format the prompt with both abstract and text
            formatted_prompt = prompt.format(abstract=self.abstract, text=text)
            
            # Generate the summary
            logger.info("Generating summary for text")
            response = self.llm.invoke(formatted_prompt)
            
            # Optional: log the model's thinking process (reasoning)
            if hasattr(response, 'additional_kwargs') and "thinking" in response.additional_kwargs:
                logger.debug(f"Model thinking process: {response.additional_kwargs['thinking']}")
            
            return response.content if hasattr(response, 'content') else response
        except Exception as e:
            logger.error(f"Error during summarization: {str(e)}")
            return f"Error summarizing section: {str(e)}"
            
    def summarize_all_sections(self) -> Dict[str, str]:
        """Summarize all sections in the document.
        
        Returns
        -------
        Dict[str, str]
            Dictionary mapping section group IDs to summaries
        """
        # Find all unique group IDs
        group_ids = set()
        for annot in self.annotations_list:
            group_ids.add(annot["group"])
        
        # Generate summaries for each group with a progress bar
        group_ids_list = sorted(list(group_ids))
        progress_bar = st.sidebar.progress(0.0)
        status_text = st.sidebar.empty()
        
        for i, group_id in enumerate(group_ids_list):
            status_text.text(f"Summarizing: {group_id}")
            summary = self.summarize_section(group_id)
            self._summaries[group_id] = summary
            # Update progress bar
            progress = (i + 1) / len(group_ids_list)
            progress_bar.progress(progress)
        
        # Clear status text after completion
        status_text.text("Summarization complete!")
        # Save all summaries to file
        self._save_summaries()
            
        return self._summaries
    

class SummaryReader:
    def __init__(self, paper_path: str):
        self.paper_path = paper_path
        self.path_handler = PathHandler(paper_path)
        self.audio_mapping_path = self.path_handler.audio_mapping_path
        self.audio_dir = self.path_handler.audio_dir
        
        # Ensure audio directory exists
        self.audio_dir.mkdir(exist_ok=True)
        
        # Load audio mapping if it exists
        self._audio_mapping = self._load_audio_mapping()

        self.voice_mapping = {
            "title": "gU0LNdkMOQCOrPrwtbee",  # Enthusiastic voice for titles
            "else": "ch0vU2DwfJVmFG2iZy89",    # Default voice for other sections
        }
    
    def _load_audio_mapping(self) -> Dict[str, str]:
        """Load audio mapping from file if it exists."""
        if self.audio_mapping_path.exists():
            try:
                with open(self.audio_mapping_path, "r") as f:
                    return json.load(f)
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

    def delete_audio_file(self, group_id: str) -> None:
        """Delete audio file for a section."""
        if group_id in self._audio_mapping:
            file_path = Path(self._audio_mapping[group_id])
            if file_path.exists():
                try:
                    logger.info(f"Deleting audio file for {group_id} from {file_path}")
                    file_path.unlink()  # Delete the file from the filesystem
                except Exception as e:
                    logger.error(f"Error deleting audio file {file_path}: {str(e)}")
            del self._audio_mapping[group_id]
            self._save_audio_mapping()
        
    def save_summary_audio_to_file(self, group_id: str, summary: str) -> str:
        """Save summary text as audio to a file.
        
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
        voice_id = self.voice_mapping["title"] if "title" in group_id else self.voice_mapping["else"]
        
        # Create filename based on section ID and voice
        paper_name = self.path_handler.name
        filename = f"{paper_name}_{group_id}_{voice_id}.mp3"
        file_path = str(self.audio_dir / filename)
        
        # Check if file already exists
        if os.path.exists(file_path):
            logger.info(f"Audio file for {group_id} already exists at {file_path}")
            
            # Make sure it's in the mapping
            if group_id not in self._audio_mapping:
                self._audio_mapping[group_id] = file_path
                self._save_audio_mapping()
                
            return file_path
        
        try:
            # Generate audio and save to file in one step
            logger.info(f"Converting summary for {group_id} to audio")
            saved_path = text_to_audio_file(summary, file_path, voice_id)
                
            # Update mapping
            self._audio_mapping[group_id] = saved_path
            self._save_audio_mapping()
            
            logger.info(f"Saved audio for {group_id} to {saved_path}")
            return saved_path
        except Exception as e:
            logger.error(f"Error saving audio for {group_id}: {str(e)}")
            return ""
    
    def get_audio_bytes(self, group_id: str, summary: str = None) -> bytes:
        """Get audio bytes for a section, generating the file if needed.
        
        Parameters
        ----------
        group_id : str
            The section ID
        summary : str, optional
            Summary text to use if audio file needs to be generated
            
        Returns
        -------
        bytes
            Audio bytes for the section
        """
        # Check if we have a cached audio file
        if group_id in self._audio_mapping:
            file_path = Path(self._audio_mapping[group_id])
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
            file_path = self.save_summary_audio_to_file(group_id, summary)
            if file_path:
                try:
                    with open(file_path, "rb") as f:
                        return f.read()
                except Exception as e:
                    logger.error(f"Error reading newly generated audio file {file_path}: {str(e)}")
        
        # Return empty bytes if all else fails
        logger.warning(f"Unable to get audio bytes for {group_id}, returning empty bytes")
        return b""
    
    def generate_all_audio(self, summaries: Dict[str, str]) -> Dict[str, str]:
        """Generate audio files for all summaries.
        
        Parameters
        ----------
        summaries : Dict[str, str]
            Dictionary mapping section IDs to summary text
            
        Returns
        -------
        Dict[str, str]
            Dictionary mapping section IDs to audio file paths
        """
        results = {}
        failed = []
        
        # Create progress bar and status elements
        total = len(summaries)
        progress_bar = st.sidebar.progress(0.0)
        status_text = st.sidebar.empty()
        
        logger.info(f"Starting batch audio generation for {total} summaries")
        
        # Process each summary
        for i, (group_id, summary) in enumerate(summaries.items()):
            try:
                # Show current progress in UI
                status_text.text(f"Generating audio: {group_id} ({i+1}/{total})")
                
                # Generate the audio file
                file_path = self.save_summary_audio_to_file(group_id, summary)
                
                if file_path:
                    results[group_id] = file_path
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
            
        return results
        
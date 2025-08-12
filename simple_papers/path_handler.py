from pathlib import Path


class PathHandler:
    """
    Handles file paths for paper documents and related files.
    """
    
    def __init__(self, path: str):
        """
        Initialize a PathHandler object.
        
        Parameters
        ----------
        path : str
            Path to the PDF file (can be absolute or relative)
        """
        # Convert to Path object but keep original string for reference
        self.original_path = path
        self.path_obj = Path(path)
        
        # Get the filename without extension
        self.name = self.path_obj.stem
        
        # Get the directory containing the file
        self.dir = self.path_obj.parent
        
        # Create paths for related files
        self.parsed_doc_path = self.dir.joinpath(f"{self.name}.pkl")
        self.annotations_path = self.dir.joinpath(f"{self.name}_annotations.json")
        self.summaries_path = self.dir.joinpath(f"{self.name}_summaries.json")
        self.audio_mapping_path = self.dir.joinpath(f"{self.name}_audio_mapping.json")
        self.audio_dir = self.dir.joinpath("audio")
        
    @property
    def path(self):
        """
        Return the original path string.
        
        Returns
        -------
        str
            Original path string
        """
        return self.original_path

    @staticmethod
    def create_path_for_uploaded_file(uploaded_file_name: str):
        """
        Create an organized path for an uploaded file in the papers directory.
        
        Parameters
        ----------
        uploaded_file_name : str
            Original name of the uploaded file
            
        Returns
        -------
        tuple
            (absolute_file_path, relative_file_path, papers_dir_path) - The absolute path where the file
            should be saved, the relative path for storage in the paper registry, and the papers directory path
        """
        # Get the project root directory
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent
        papers_dir = project_root.joinpath('papers')
        
        # Ensure papers directory exists
        papers_dir.mkdir(exist_ok=True)
        
        # Create a clean filename (remove any problematic characters)
        clean_name = Path(uploaded_file_name).stem
        
        # Create a directory for this specific paper
        paper_dir = papers_dir.joinpath(clean_name)
        paper_dir.mkdir(exist_ok=True)
        
        # Full absolute path for the PDF file
        abs_pdf_path = paper_dir.joinpath(f"{clean_name}.pdf")
        
        # Create the relative path for storage in registry
        # Format: papers/paper_name/paper_name.pdf
        rel_pdf_path = f"papers/{clean_name}/{clean_name}.pdf"
        
        return abs_pdf_path, rel_pdf_path, papers_dir

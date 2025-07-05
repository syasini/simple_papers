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
            Path to the PDF file
        """
        self.path = path
        self.name = self.get_name_from_path(path)
        self.dir = self.get_dir_from_path(path)

        self.parsed_doc_path = self.dir.joinpath(f"{self.name}.pkl")
        self.annotations_path = self.dir.joinpath(f"{self.name}_annotations.json")
        self.summaries_path = self.dir.joinpath(f"{self.name}_summaries.json")
        self.audio_mapping_path = self.dir.joinpath(f"{self.name}_audio_mapping.json")
        self.audio_dir = self.dir.joinpath("audio")
        
    def get_name_from_path(self, path):
        """
        Extract the base filename without extension from a path.
        
        Parameters
        ----------
        path : str
            File path
            
        Returns
        -------
        str
            Base filename without extension
        """
        return Path(path).stem.lower()
    
    def get_dir_from_path(self, path):
        """
        Extract the directory part from a path.
        
        Parameters
        ----------
        path : str
            File path
            
        Returns
        -------
        Path
            Directory path
        """
        return Path(path).parent

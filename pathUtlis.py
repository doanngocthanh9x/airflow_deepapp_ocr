# c:\Automation\Airflow\pathUtlis.py
"""
Path Utilities - Manage project paths and directory navigation
"""
from pathlib import Path
from typing import Union

class PathManager:
    """Centralized path management for the Airflow project"""
    
    # Get root project path (where docker-compose.yaml is)
    PROJECT_ROOT = Path(__file__).parent.resolve()
    
    # Define all major directories relative to PROJECT_ROOT
    DIRS = {
        'root': PROJECT_ROOT,
        'dags': PROJECT_ROOT / 'dags',
        'services': PROJECT_ROOT / 'services',
        'webapp': PROJECT_ROOT / 'webapp',
        'models': PROJECT_ROOT / 'models',
        'config': PROJECT_ROOT / 'config',
        'logs': PROJECT_ROOT / 'logs',
        'data': PROJECT_ROOT / 'dags' / 'data',
        'test_img': PROJECT_ROOT / 'test_img',
        'docs': PROJECT_ROOT / 'docs',
        'plugins': PROJECT_ROOT / 'plugins',
    }
    
    @classmethod
    def get_root(cls) -> Path:
        """Get project root path"""
        return cls.PROJECT_ROOT
    
    @classmethod
    def get_path(cls, folder_name: str) -> Path:
        """
        Get path to a specific folder
        
        Args:
            folder_name: Name of the folder (key from DIRS dict)
            
        Returns:
            Path object to the folder
            
        Raises:
            KeyError: If folder_name not found in DIRS
        """
        if folder_name not in cls.DIRS:
            raise KeyError(
                f"Unknown folder: {folder_name}\n"
                f"Available folders: {', '.join(cls.DIRS.keys())}"
            )
        return cls.DIRS[folder_name]
    
    @classmethod
    def get_file(cls, folder_name: str, filename: str) -> Path:
        """
        Get path to a specific file in a folder
        
        Args:
            folder_name: Name of the folder
            filename: Name of the file
            
        Returns:
            Path object to the file
        """
        folder_path = cls.get_path(folder_name)
        return folder_path / filename
    
    @classmethod
    def list_folders(cls) -> dict:
        """Get all available folders"""
        return {key: str(path) for key, path in cls.DIRS.items()}
    
    @classmethod
    def ensure_folder_exists(cls, folder_name: str) -> Path:
        """
        Ensure a folder exists, create if not
        
        Args:
            folder_name: Name of the folder
            
        Returns:
            Path object to the folder
        """
        folder_path = cls.get_path(folder_name)
        folder_path.mkdir(parents=True, exist_ok=True)
        return folder_path
    
    @classmethod
    def get_relative_path(cls, file_path: Union[str, Path], relative_to: str = 'root') -> Path:
        """
        Get relative path from a reference folder
        
        Args:
            file_path: Path to convert
            relative_to: Reference folder (default: 'root')
            
        Returns:
            Relative path
        """
        file_path = Path(file_path)
        reference_path = cls.get_path(relative_to)
        try:
            return file_path.relative_to(reference_path)
        except ValueError:
            return file_path


# Convenience functions for quick access
def get_root() -> Path:
    """Get project root"""
    return PathManager.get_root()

def get_path(folder_name: str) -> Path:
    """Get path to folder"""
    return PathManager.get_path(folder_name)

def get_file(folder_name: str, filename: str) -> Path:
    """Get path to file"""
    return PathManager.get_file(folder_name, filename)

def list_folders() -> dict:
    """List all available folders"""
    return PathManager.list_folders()

def ensure_folder(folder_name: str) -> Path:
    """Ensure folder exists"""
    return PathManager.ensure_folder_exists(folder_name)


# Example usage
if __name__ == "__main__":
    print("=== Project Path Structure ===\n")
    
    # Show root
    print(f"Project Root: {get_root()}\n")
    
    # Show all folders
    print("All Folders:")
    for name, path in list_folders().items():
        print(f"  {name:15} -> {path}")
    
    print("\n=== Example Usage ===")
    print(f"DAGs folder: {get_path('dags')}")
    print(f"Services folder: {get_path('services')}")
    print(f"Config file: {get_file('config', 'airflow.cfg')}")
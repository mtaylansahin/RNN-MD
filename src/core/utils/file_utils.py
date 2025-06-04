"""File management utilities for safe file operations."""

import os
import shutil
import json
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
from dataclasses import dataclass
import tempfile
from contextlib import contextmanager

from .logging_utils import get_logger


logger = get_logger(__name__)


@dataclass
class FileOperationResult:
    """Result of a file operation."""
    
    success: bool
    source_path: Optional[str] = None
    destination_path: Optional[str] = None
    error_message: Optional[str] = None
    files_processed: int = 0


class FileOperationError(Exception):
    """Exception raised when file operations fail."""
    
    def __init__(self, message: str, path: Optional[str] = None):
        self.path = path
        super().__init__(message)


class FileManager:
    """Manages file operations with proper error handling and logging."""
    
    def __init__(self):
        """Initialize the file manager."""
        self.logger = get_logger(__name__)
    
    def copy_files_by_extension(
        self,
        source_directory: Union[str, Path],
        destination_directory: Union[str, Path],
        extensions: List[str],
        overwrite: bool = False
    ) -> FileOperationResult:
        """Copy files with specified extensions from source to destination.
        
        Args:
            source_directory: Source directory path
            destination_directory: Destination directory path
            extensions: List of file extensions to copy (e.g., ['.txt', '.py'])
            overwrite: Whether to overwrite existing files
            
        Returns:
            FileOperationResult with operation details
        """
        source_path = Path(source_directory)
        destination_path = Path(destination_directory)
        
        if not source_path.exists():
            error_msg = f"Source directory does not exist: {source_path}"
            self.logger.error(error_msg)
            return FileOperationResult(
                success=False,
                source_path=str(source_path),
                error_message=error_msg
            )
        
        try:
            # Create destination directory
            destination_path.mkdir(parents=True, exist_ok=True)
            
            files_copied = 0
            extensions_set = set(ext.lower() for ext in extensions)
            
            for file_path in source_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in extensions_set:
                    dest_file = destination_path / file_path.name
                    
                    if dest_file.exists() and not overwrite:
                        self.logger.warning(f"Skipping existing file: {dest_file}")
                        continue
                    
                    shutil.copy2(file_path, dest_file)
                    self.logger.info(f"Copied {file_path} to {dest_file}")
                    files_copied += 1
            
            self.logger.info(f"Successfully copied {files_copied} files")
            return FileOperationResult(
                success=True,
                source_path=str(source_path),
                destination_path=str(destination_path),
                files_processed=files_copied
            )
            
        except Exception as e:
            error_msg = f"Failed to copy files: {e}"
            self.logger.error(error_msg)
            return FileOperationResult(
                success=False,
                source_path=str(source_path),
                destination_path=str(destination_path),
                error_message=error_msg
            )
    
    def copy_specific_files(
        self,
        source_directory: Union[str, Path],
        destination_directory: Union[str, Path],
        filenames: List[str],
        overwrite: bool = False
    ) -> FileOperationResult:
        """Copy specific files from source to destination.
        
        Args:
            source_directory: Source directory path
            destination_directory: Destination directory path
            filenames: List of specific filenames to copy
            overwrite: Whether to overwrite existing files
            
        Returns:
            FileOperationResult with operation details
        """
        source_path = Path(source_directory)
        destination_path = Path(destination_directory)
        
        if not source_path.exists():
            error_msg = f"Source directory does not exist: {source_path}"
            self.logger.error(error_msg)
            return FileOperationResult(
                success=False,
                source_path=str(source_path),
                error_message=error_msg
            )
        
        try:
            # Create destination directory
            destination_path.mkdir(parents=True, exist_ok=True)
            
            files_copied = 0
            missing_files = []
            
            for filename in filenames:
                source_file = source_path / filename
                dest_file = destination_path / filename
                
                if not source_file.exists():
                    missing_files.append(filename)
                    continue
                
                if dest_file.exists() and not overwrite:
                    self.logger.warning(f"Skipping existing file: {dest_file}")
                    continue
                
                shutil.copy2(source_file, dest_file)
                self.logger.info(f"Copied {source_file} to {dest_file}")
                files_copied += 1
            
            if missing_files:
                self.logger.warning(f"Files not found: {missing_files}")
            
            self.logger.info(f"Successfully copied {files_copied} files")
            return FileOperationResult(
                success=True,
                source_path=str(source_path),
                destination_path=str(destination_path),
                files_processed=files_copied
            )
            
        except Exception as e:
            error_msg = f"Failed to copy files: {e}"
            self.logger.error(error_msg)
            return FileOperationResult(
                success=False,
                source_path=str(source_path),
                destination_path=str(destination_path),
                error_message=error_msg
            )
    
    def move_file(
        self,
        source_path: Union[str, Path],
        destination_path: Union[str, Path],
        create_dirs: bool = True
    ) -> FileOperationResult:
        """Move a file from source to destination.
        
        Args:
            source_path: Source file path
            destination_path: Destination file path
            create_dirs: Whether to create destination directories
            
        Returns:
            FileOperationResult with operation details
        """
        source = Path(source_path)
        destination = Path(destination_path)
        
        if not source.exists():
            error_msg = f"Source file does not exist: {source}"
            self.logger.error(error_msg)
            return FileOperationResult(
                success=False,
                source_path=str(source),
                error_message=error_msg
            )
        
        try:
            if create_dirs:
                destination.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.move(str(source), str(destination))
            self.logger.info(f"Moved {source} to {destination}")
            
            return FileOperationResult(
                success=True,
                source_path=str(source),
                destination_path=str(destination),
                files_processed=1
            )
            
        except Exception as e:
            error_msg = f"Failed to move file: {e}"
            self.logger.error(error_msg)
            return FileOperationResult(
                success=False,
                source_path=str(source),
                destination_path=str(destination),
                error_message=error_msg
            )
    
    def delete_file(self, file_path: Union[str, Path]) -> FileOperationResult:
        """Delete a file safely.
        
        Args:
            file_path: Path to file to delete
            
        Returns:
            FileOperationResult with operation details
        """
        path = Path(file_path)
        
        if not path.exists():
            self.logger.warning(f"File does not exist: {path}")
            return FileOperationResult(
                success=True,
                source_path=str(path),
                files_processed=0
            )
        
        try:
            path.unlink()
            self.logger.info(f"Deleted file: {path}")
            
            return FileOperationResult(
                success=True,
                source_path=str(path),
                files_processed=1
            )
            
        except Exception as e:
            error_msg = f"Failed to delete file: {e}"
            self.logger.error(error_msg)
            return FileOperationResult(
                success=False,
                source_path=str(path),
                error_message=error_msg
            )
    
    def create_directory(
        self,
        directory_path: Union[str, Path],
        parents: bool = True,
        exist_ok: bool = True
    ) -> FileOperationResult:
        """Create a directory safely.
        
        Args:
            directory_path: Path to directory to create
            parents: Whether to create parent directories
            exist_ok: Whether to ignore if directory already exists
            
        Returns:
            FileOperationResult with operation details
        """
        path = Path(directory_path)
        
        try:
            path.mkdir(parents=parents, exist_ok=exist_ok)
            self.logger.info(f"Created directory: {path}")
            
            return FileOperationResult(
                success=True,
                destination_path=str(path),
                files_processed=1
            )
            
        except Exception as e:
            error_msg = f"Failed to create directory: {e}"
            self.logger.error(error_msg)
            return FileOperationResult(
                success=False,
                destination_path=str(path),
                error_message=error_msg
            )
    
    def read_json_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Read JSON file safely.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Dictionary containing JSON data
            
        Raises:
            FileOperationError: If file cannot be read or parsed
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileOperationError(f"JSON file does not exist: {path}", str(path))
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            self.logger.debug(f"Read JSON file: {path}")
            return data
            
        except json.JSONDecodeError as e:
            raise FileOperationError(f"Invalid JSON format in {path}: {e}", str(path))
        except Exception as e:
            raise FileOperationError(f"Failed to read JSON file {path}: {e}", str(path))
    
    def write_json_file(
        self,
        data: Dict[str, Any],
        file_path: Union[str, Path],
        indent: int = 2,
        create_dirs: bool = True
    ) -> FileOperationResult:
        """Write data to JSON file safely.
        
        Args:
            data: Data to write to JSON file
            file_path: Path to JSON file
            indent: JSON indentation level
            create_dirs: Whether to create parent directories
            
        Returns:
            FileOperationResult with operation details
        """
        path = Path(file_path)
        
        try:
            if create_dirs:
                path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w') as f:
                json.dump(data, f, indent=indent)
            
            self.logger.info(f"Wrote JSON file: {path}")
            return FileOperationResult(
                success=True,
                destination_path=str(path),
                files_processed=1
            )
            
        except Exception as e:
            error_msg = f"Failed to write JSON file: {e}"
            self.logger.error(error_msg)
            return FileOperationResult(
                success=False,
                destination_path=str(path),
                error_message=error_msg
            )
    
    def write_metadata_file(
        self,
        metadata: Dict[str, Any],
        file_path: Union[str, Path],
        create_dirs: bool = True
    ) -> FileOperationResult:
        """Write metadata to a text file.
        
        Args:
            metadata: Metadata dictionary to write
            file_path: Path to metadata file
            create_dirs: Whether to create parent directories
            
        Returns:
            FileOperationResult with operation details
        """
        path = Path(file_path)
        
        try:
            if create_dirs:
                path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w') as f:
                for key, value in metadata.items():
                    f.write(f"{key}: {value}\n")
            
            self.logger.info(f"Wrote metadata file: {path}")
            return FileOperationResult(
                success=True,
                destination_path=str(path),
                files_processed=1
            )
            
        except Exception as e:
            error_msg = f"Failed to write metadata file: {e}"
            self.logger.error(error_msg)
            return FileOperationResult(
                success=False,
                destination_path=str(path),
                error_message=error_msg
            )
    
    @contextmanager
    def temporary_directory(self, suffix: Optional[str] = None):
        """Create a temporary directory context manager.
        
        Args:
            suffix: Optional suffix for temporary directory name
            
        Yields:
            Path object of the temporary directory
        """
        temp_dir = tempfile.mkdtemp(suffix=suffix)
        try:
            self.logger.debug(f"Created temporary directory: {temp_dir}")
            yield Path(temp_dir)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            self.logger.debug(f"Cleaned up temporary directory: {temp_dir}")
    
    def validate_file_exists(self, file_path: Union[str, Path]) -> bool:
        """Validate that a file exists.
        
        Args:
            file_path: Path to file to check
            
        Returns:
            True if file exists, False otherwise
        """
        path = Path(file_path)
        exists = path.exists() and path.is_file()
        
        if not exists:
            self.logger.warning(f"File does not exist: {path}")
        
        return exists
    
    def get_file_size(self, file_path: Union[str, Path]) -> int:
        """Get file size in bytes.
        
        Args:
            file_path: Path to file
            
        Returns:
            File size in bytes
            
        Raises:
            FileOperationError: If file does not exist
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileOperationError(f"File does not exist: {path}", str(path))
        
        return path.stat().st_size 
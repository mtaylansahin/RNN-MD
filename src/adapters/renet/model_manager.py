"""Model manager for handling RE-Net model artifacts and metadata."""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Add src to path for imports if not already there
src_path = Path(__file__).parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from core.config import ExperimentConfig
from core.utils import FileManager, get_logger


logger = get_logger(__name__)


class RENetModelManager:
    """Manages RE-Net model files, checkpoints, and metadata."""
    
    def __init__(self, config: ExperimentConfig):
        """Initialize model manager.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.file_manager = FileManager()
        
        # Model file patterns used by RE-Net
        self.model_patterns = {
            "global_model": f"max{config.maxpool}rgcn_global.pth",
            "global_model2": f"max{config.maxpool}rgcn_global2.pth", 
            "main_model": "rgcn.pth",
            "backup_model": "rgcn_backup.pth",
            "graph_data": "rgcn_graph.pth"
        }
    
    @property
    def models_directory(self) -> Path:
        """Get the models directory path."""
        return Path(self.config.renet_directory) / "models" / self.config.experiment_name
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about available model files.
        
        Returns:
            Dictionary with model file information
        """
        info = {}
        models_dir = self.models_directory
        
        if not models_dir.exists():
            self.logger.warning(f"Models directory does not exist: {models_dir}")
            return info
        
        for model_type, filename in self.model_patterns.items():
            model_path = models_dir / filename
            
            if model_path.exists():
                stat = model_path.stat()
                info[model_type] = {
                    "path": str(model_path),
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "exists": True
                }
            else:
                info[model_type] = {
                    "path": str(model_path),
                    "exists": False
                }
        
        return info
    
    def backup_models(self, backup_suffix: Optional[str] = None) -> bool:
        """Create backup copies of current model files.
        
        Args:
            backup_suffix: Optional suffix to add to backup files
            
        Returns:
            True if backup successful, False otherwise
        """
        if backup_suffix is None:
            backup_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        models_dir = self.models_directory
        backup_dir = models_dir.parent / f"backup_{self.config.experiment_name}_{backup_suffix}"
        
        try:
            # Create backup directory
            result = self.file_manager.create_directory(backup_dir)
            if not result.success:
                self.logger.error(f"Failed to create backup directory: {result.error_message}")
                return False
            
            # Copy model files to backup
            backed_up_files = 0
            for model_type, filename in self.model_patterns.items():
                source_path = models_dir / filename
                
                if source_path.exists():
                    destination_path = backup_dir / filename
                    copy_result = self.file_manager.copy_specific_files(
                        source_directory=models_dir,
                        destination_directory=backup_dir,
                        filenames=[filename]
                    )
                    
                    if copy_result.success:
                        backed_up_files += 1
                        self.logger.info(f"Backed up {model_type}: {filename}")
                    else:
                        self.logger.error(f"Failed to backup {model_type}: {copy_result.error_message}")
            
            if backed_up_files > 0:
                self.logger.info(f"Successfully backed up {backed_up_files} model files to {backup_dir}")
                return True
            else:
                self.logger.warning("No model files found to backup")
                return False
                
        except Exception as e:
            self.logger.error(f"Error during model backup: {e}")
            return False
    
    def restore_models(self, backup_directory: str) -> bool:
        """Restore model files from backup.
        
        Args:
            backup_directory: Path to backup directory
            
        Returns:
            True if restore successful, False otherwise
        """
        backup_path = Path(backup_directory)
        models_dir = self.models_directory
        
        if not backup_path.exists():
            self.logger.error(f"Backup directory does not exist: {backup_path}")
            return False
        
        try:
            # Ensure models directory exists
            result = self.file_manager.create_directory(models_dir)
            if not result.success:
                self.logger.error(f"Failed to create models directory: {result.error_message}")
                return False
            
            # Restore model files
            restored_files = 0
            for model_type, filename in self.model_patterns.items():
                backup_file = backup_path / filename
                
                if backup_file.exists():
                    copy_result = self.file_manager.copy_specific_files(
                        source_directory=backup_path,
                        destination_directory=models_dir,
                        filenames=[filename],
                        overwrite=True
                    )
                    
                    if copy_result.success:
                        restored_files += 1
                        self.logger.info(f"Restored {model_type}: {filename}")
                    else:
                        self.logger.error(f"Failed to restore {model_type}: {copy_result.error_message}")
            
            if restored_files > 0:
                self.logger.info(f"Successfully restored {restored_files} model files from {backup_path}")
                return True
            else:
                self.logger.warning("No model files found in backup to restore")
                return False
                
        except Exception as e:
            self.logger.error(f"Error during model restoration: {e}")
            return False
    
    def clean_old_models(self, keep_latest: int = 3) -> bool:
        """Clean up old model backups, keeping only the latest N backups.
        
        Args:
            keep_latest: Number of latest backups to keep
            
        Returns:
            True if cleanup successful, False otherwise
        """
        try:
            models_parent_dir = self.models_directory.parent
            backup_pattern = f"backup_{self.config.experiment_name}_*"
            
            # Find all backup directories
            backup_dirs = list(models_parent_dir.glob(backup_pattern))
            
            if len(backup_dirs) <= keep_latest:
                self.logger.info(f"Found {len(backup_dirs)} backups, no cleanup needed")
                return True
            
            # Sort by modification time (newest first)
            backup_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            
            # Remove old backups
            dirs_to_remove = backup_dirs[keep_latest:]
            removed_count = 0
            
            for backup_dir in dirs_to_remove:
                try:
                    import shutil
                    shutil.rmtree(backup_dir)
                    self.logger.info(f"Removed old backup: {backup_dir}")
                    removed_count += 1
                except Exception as e:
                    self.logger.error(f"Failed to remove backup {backup_dir}: {e}")
            
            self.logger.info(f"Cleaned up {removed_count} old backup directories")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during backup cleanup: {e}")
            return False
    
    def validate_model_completeness(self) -> Dict[str, bool]:
        """Validate that all required model files are present.
        
        Returns:
            Dictionary mapping model types to availability status
        """
        validation_results = {}
        models_dir = self.models_directory
        
        # Core required models for RE-Net operation
        required_models = ["global_model", "main_model"]
        
        for model_type, filename in self.model_patterns.items():
            model_path = models_dir / filename
            exists = model_path.exists()
            validation_results[model_type] = exists
            
            if model_type in required_models:
                if exists:
                    self.logger.info(f"Required model found: {model_type}")
                else:
                    self.logger.error(f"Required model missing: {model_type} at {model_path}")
        
        return validation_results
    
    def get_model_storage_usage(self) -> Dict[str, float]:
        """Get storage usage information for model files.
        
        Returns:
            Dictionary with storage usage in MB
        """
        usage_info = {}
        total_size = 0.0
        
        for model_type, filename in self.model_patterns.items():
            model_path = self.models_directory / filename
            
            if model_path.exists():
                size_mb = model_path.stat().st_size / (1024 * 1024)
                usage_info[model_type] = round(size_mb, 2)
                total_size += size_mb
            else:
                usage_info[model_type] = 0.0
        
        usage_info["total"] = round(total_size, 2)
        return usage_info
    
    def archive_experiment_models(self, archive_directory: str) -> bool:
        """Archive all model files for long-term storage.
        
        Args:
            archive_directory: Directory to store archived models
            
        Returns:
            True if archiving successful, False otherwise
        """
        archive_path = Path(archive_directory)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_archive = archive_path / f"{self.config.experiment_name}_{timestamp}"
        
        try:
            # Create archive directory
            result = self.file_manager.create_directory(experiment_archive)
            if not result.success:
                self.logger.error(f"Failed to create archive directory: {result.error_message}")
                return False
            
            # Copy all model files and directories
            models_dir = self.models_directory
            if not models_dir.exists():
                self.logger.warning(f"No models directory to archive: {models_dir}")
                return False
            
            # Copy the entire models directory
            import shutil
            shutil.copytree(models_dir, experiment_archive / "models")
            
            # Create archive metadata
            metadata = {
                "experiment_name": self.config.experiment_name,
                "archive_date": datetime.now().isoformat(),
                "original_path": str(models_dir),
                "config": {
                    "gpu_device": self.config.gpu_device,
                    "sequence_length": self.config.sequence_length,
                    "maxpool": self.config.maxpool,
                    "model_type": self.config.model_type
                }
            }
            
            metadata_file = experiment_archive / "archive_metadata.txt"
            self.file_manager.write_metadata_file(metadata, metadata_file)
            
            self.logger.info(f"Successfully archived experiment models to {experiment_archive}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during model archiving: {e}")
            return False 
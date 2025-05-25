"""Main RE-Net adapter providing clean interface to RE-Net functionality."""

import os
import random
import string
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

# Add src to path for imports if not already there
src_path = Path(__file__).parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from core.config import ExperimentConfig
from core.utils import FileManager, get_logger
from core.utils.process_utils import RENetProcessManager, ProcessResult, ProcessError
from .model_manager import RENetModelManager


logger = get_logger(__name__)


@dataclass
class TrainingResult:
    """Result of a training operation."""
    
    success: bool
    phase: str
    hyperparameters: Dict
    execution_time: float
    model_path: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class TestingResult:
    """Result of a testing operation."""
    
    success: bool
    output_file: Optional[str] = None
    metadata_file: Optional[str] = None
    predictions: Optional[str] = None
    execution_time: float = 0.0
    error_message: Optional[str] = None


class RENetAdapter:
    """Adapter for interfacing with RE-Net while maintaining clean separation."""
    
    def __init__(self, config: ExperimentConfig):
        """Initialize RE-Net adapter.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Initialize utilities
        self.file_manager = FileManager()
        self.process_manager = RENetProcessManager(
            renet_directory=config.renet_directory,
            timeout=7200.0  # 2 hours default timeout
        )
        self.model_manager = RENetModelManager(config)
        
        # Validate RE-Net directory
        self._validate_renet_installation()
    
    def _validate_renet_installation(self) -> None:
        """Validate that RE-Net is properly installed and accessible."""
        renet_path = Path(self.config.renet_directory)
        
        required_files = ["pretrain.py", "train.py", "test.py"]
        missing_files = []
        
        for file_name in required_files:
            if not (renet_path / file_name).exists():
                missing_files.append(file_name)
        
        if missing_files:
            raise FileNotFoundError(
                f"RE-Net installation incomplete. Missing files: {missing_files}"
            )
        
        self.logger.info(f"RE-Net installation validated at: {renet_path}")
    
    def prepare_dataset(self) -> bool:
        """Prepare dataset for RE-Net training.
        
        Returns:
            True if dataset preparation successful, False otherwise
        """
        try:
            self.logger.info("Preparing dataset for RE-Net")
            
            # Copy required files to RE-Net data directory
            required_files = [
                "train.txt", "valid.txt", "test.txt", "stat.txt", "get_history_graph.py"
            ]
            
            result = self.file_manager.copy_specific_files(
                source_directory=self.config.data_config.data_directory,
                destination_directory=self.config.renet_data_directory,
                filenames=required_files,
                overwrite=True
            )
            
            if not result.success:
                self.logger.error(f"Failed to copy dataset files: {result.error_message}")
                return False
            
            # Generate history graphs
            self.logger.info("Generating history graphs")
            graph_result = self.process_manager.generate_history_graph(
                self.config.experiment_name
            )
            
            if not graph_result.succeeded:
                self.logger.error(f"History graph generation failed: {graph_result.stderr}")
                return False
            
            self.logger.info("Dataset preparation completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Dataset preparation failed: {e}")
            return False
    
    def run_pretraining(
        self,
        dropout: float,
        n_hidden: int,
        learning_rate: float,
        max_epochs: int,
        batch_size: int
    ) -> TrainingResult:
        """Run RE-Net pretraining phase.
        
        Args:
            dropout: Dropout rate
            n_hidden: Number of hidden units
            learning_rate: Learning rate
            max_epochs: Maximum epochs
            batch_size: Batch size
            
        Returns:
            TrainingResult with operation details
        """
        hyperparameters = {
            "dropout": dropout,
            "n_hidden": n_hidden,
            "learning_rate": learning_rate,
            "max_epochs": max_epochs,
            "batch_size": batch_size
        }
        
        self.logger.info(f"Starting pretraining with parameters: {hyperparameters}")
        
        try:
            result = self.process_manager.pretrain_model(
                dataset_name=self.config.experiment_name,
                dropout=dropout,
                n_hidden=n_hidden,
                learning_rate=learning_rate,
                max_epochs=max_epochs,
                batch_size=batch_size,
                gpu_device=self.config.gpu_device,
                maxpool=self.config.maxpool
            )
            
            if result.succeeded:
                self.logger.info(f"Pretraining completed in {result.execution_time:.2f}s")
                return TrainingResult(
                    success=True,
                    phase="pretraining",
                    hyperparameters=hyperparameters,
                    execution_time=result.execution_time
                )
            else:
                self.logger.error(f"Pretraining failed: {result.stderr}")
                return TrainingResult(
                    success=False,
                    phase="pretraining",
                    hyperparameters=hyperparameters,
                    execution_time=result.execution_time,
                    error_message=result.stderr
                )
                
        except ProcessError as e:
            self.logger.error(f"Pretraining process failed: {e}")
            return TrainingResult(
                success=False,
                phase="pretraining",
                hyperparameters=hyperparameters,
                execution_time=0.0,
                error_message=str(e)
            )
        except Exception as e:
            self.logger.error(f"Unexpected error during pretraining: {e}")
            return TrainingResult(
                success=False,
                phase="pretraining",
                hyperparameters=hyperparameters,
                execution_time=0.0,
                error_message=str(e)
            )
    
    def run_training(
        self,
        dropout: float,
        n_hidden: int,
        learning_rate: float,
        max_epochs: int,
        batch_size: int
    ) -> TrainingResult:
        """Run RE-Net training phase.
        
        Args:
            dropout: Dropout rate
            n_hidden: Number of hidden units
            learning_rate: Learning rate
            max_epochs: Maximum epochs
            batch_size: Batch size
            
        Returns:
            TrainingResult with operation details
        """
        hyperparameters = {
            "dropout": dropout,
            "n_hidden": n_hidden,
            "learning_rate": learning_rate,
            "max_epochs": max_epochs,
            "batch_size": batch_size
        }
        
        self.logger.info(f"Starting training with parameters: {hyperparameters}")
        
        try:
            result = self.process_manager.train_model(
                dataset_name=self.config.experiment_name,
                dropout=dropout,
                n_hidden=n_hidden,
                learning_rate=learning_rate,
                max_epochs=max_epochs,
                batch_size=batch_size,
                gpu_device=self.config.gpu_device,
                num_k=self.config.num_k_parameter
            )
            
            if result.succeeded:
                self.logger.info(f"Training completed in {result.execution_time:.2f}s")
                return TrainingResult(
                    success=True,
                    phase="training",
                    hyperparameters=hyperparameters,
                    execution_time=result.execution_time
                )
            else:
                self.logger.error(f"Training failed: {result.stderr}")
                return TrainingResult(
                    success=False,
                    phase="training",
                    hyperparameters=hyperparameters,
                    execution_time=result.execution_time,
                    error_message=result.stderr
                )
                
        except ProcessError as e:
            self.logger.error(f"Training process failed: {e}")
            return TrainingResult(
                success=False,
                phase="training",
                hyperparameters=hyperparameters,
                execution_time=0.0,
                error_message=str(e)
            )
        except Exception as e:
            self.logger.error(f"Unexpected error during training: {e}")
            return TrainingResult(
                success=False,
                phase="training",
                hyperparameters=hyperparameters,
                execution_time=0.0,
                error_message=str(e)
            )
    
    def run_testing(self, n_hidden: int, run_id: str) -> TestingResult:
        """Run RE-Net testing phase.
        
        Args:
            n_hidden: Number of hidden units
            run_id: Unique identifier for this test run
            
        Returns:
            TestingResult with operation details
        """
        self.logger.info(f"Starting testing with n_hidden={n_hidden}, run_id={run_id}")
        
        try:
            result = self.process_manager.test_model(
                dataset_name=self.config.experiment_name,
                n_hidden=n_hidden,
                gpu_device=self.config.gpu_device,
                num_k=self.config.num_k_parameter
            )
            
            if result.succeeded:
                # Handle output file management
                output_file, metadata_file = self._handle_test_outputs(run_id)
                
                self.logger.info(f"Testing completed in {result.execution_time:.2f}s")
                return TestingResult(
                    success=True,
                    output_file=output_file,
                    metadata_file=metadata_file,
                    execution_time=result.execution_time
                )
            else:
                self.logger.error(f"Testing failed: {result.stderr}")
                return TestingResult(
                    success=False,
                    execution_time=result.execution_time,
                    error_message=result.stderr
                )
                
        except ProcessError as e:
            self.logger.error(f"Testing process failed: {e}")
            return TestingResult(
                success=False,
                execution_time=0.0,
                error_message=str(e)
            )
        except Exception as e:
            self.logger.error(f"Unexpected error during testing: {e}")
            return TestingResult(
                success=False,
                execution_time=0.0,
                error_message=str(e)
            )
    
    def _handle_test_outputs(self, run_id: str) -> Tuple[Optional[str], Optional[str]]:
        """Handle test output files and move them to appropriate locations.
        
        Args:
            run_id: Unique identifier for this test run
            
        Returns:
            Tuple of (output_file_path, metadata_file_path)
        """
        try:
            results_directory = self.config.get_results_directory(run_id)
            
            # Expected output file from RE-Net testing
            expected_output = f"{self.config.experiment_name}_prediction_set_1.txt"
            source_output_path = Path(self.config.renet_directory) / expected_output
            
            if source_output_path.exists():
                # Move output file to results directory
                final_output_file = f"{self.config.experiment_name}_prediction_set_{run_id}.txt"
                destination_path = Path(results_directory) / final_output_file
                
                move_result = self.file_manager.move_file(
                    source_path=source_output_path,
                    destination_path=destination_path
                )
                
                if move_result.success:
                    return str(destination_path), None
                else:
                    self.logger.error(f"Failed to move output file: {move_result.error_message}")
                    return None, None
            else:
                self.logger.warning(f"Expected output file not found: {source_output_path}")
                return None, None
                
        except Exception as e:
            self.logger.error(f"Error handling test outputs: {e}")
            return None, None
    
    def create_metadata_file(
        self,
        hyperparameters: Dict,
        run_id: str,
        results_directory: str,
        pretraining_epochs: int
    ) -> Optional[str]:
        """Create metadata file for experimental run.
        
        Args:
            hyperparameters: Dictionary of hyperparameters used
            run_id: Unique identifier for this run
            results_directory: Directory to save metadata
            pretraining_epochs: Number of pretraining epochs used
            
        Returns:
            Path to created metadata file, or None if creation failed
        """
        try:
            metadata = {
                "experiment_name": self.config.experiment_name,
                "run_id": run_id,
                "data_directory": self.config.data_config.data_directory,
                "replica": self.config.data_config.replica,
                "chain1": self.config.data_config.chain1,
                "chain2": self.config.data_config.chain2,
                "train_ratio": self.config.data_config.train_ratio,
                "validation_ratio": self.config.data_config.validation_ratio,
                "pretraining_epochs": pretraining_epochs,
                **hyperparameters
            }
            
            metadata_file = Path(results_directory) / f"{self.config.experiment_name}_metadata_{run_id}.txt"
            
            result = self.file_manager.write_metadata_file(
                metadata=metadata,
                file_path=metadata_file
            )
            
            if result.success:
                return str(metadata_file)
            else:
                self.logger.error(f"Failed to create metadata file: {result.error_message}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error creating metadata file: {e}")
            return None
    
    def generate_run_id(self, length: int = 5) -> str:
        """Generate a unique run identifier.
        
        Args:
            length: Length of the generated ID
            
        Returns:
            Unique run identifier string
        """
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
    
    def cleanup_temporary_files(self) -> None:
        """Clean up temporary files created during operations."""
        try:
            # Clean up any temporary files in RE-Net directory
            renet_path = Path(self.config.renet_directory)
            temp_patterns = ["*.tmp", "*.temp", "*_backup.pth"]
            
            for pattern in temp_patterns:
                for temp_file in renet_path.glob(pattern):
                    if temp_file.is_file():
                        self.file_manager.delete_file(temp_file)
            
            self.logger.info("Temporary file cleanup completed")
            
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about saved models.
        
        Returns:
            Dictionary with model information
        """
        return self.model_manager.get_model_info() 
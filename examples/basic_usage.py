#!/usr/bin/env python3
"""Basic usage example for RNN-MD refactored architecture.

This example demonstrates how to use the refactored RNN-MD architecture
programmatically rather than through command-line interfaces.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.config import DataConfig, HyperparameterConfig, ExperimentConfig
from core.utils import setup_logging, ExperimentLogger
from adapters.renet import RENetAdapter


def create_example_config() -> ExperimentConfig:
    """Create an example configuration for demonstration."""
    
    # Define data configuration
    data_config = DataConfig(
        data_directory="/path/to/your/data",  # TODO: Update this path to your actual data directory
        replica="replica1",
        chain1="A",
        chain2="B",
        train_ratio=0.7,
        validation_ratio=0.2,
        interaction_type="residue"
    )
    
    # Define hyperparameters to explore
    hyperparameters = HyperparameterConfig(
        dropout_rates=[0.5],
        learning_rates=[0.001],
        batch_sizes=[128],
        pretrain_epochs=[20],
        train_epochs=[10],
        hidden_units=[100]
    )
    
    # Create experiment configuration
    config = ExperimentConfig(
        experiment_name="example_experiment",
        data_config=data_config,
        hyperparameters=hyperparameters,
        gpu_device=0,
        random_seed=999
    )
    
    return config


def run_simple_experiment():
    """Run a simple experiment using the refactored architecture."""
    
    # Setup logging
    setup_logging(
        log_level="INFO",
        log_file="logs/example_experiment.log",
        experiment_name="example_experiment"
    )
    
    # Create configuration
    config = create_example_config()
    
    # Initialize experiment logger
    logger = ExperimentLogger("example_experiment")
    
    # Log experiment start
    logger.log_experiment_start({
        "experiment_name": config.experiment_name,
        "data_directory": config.data_config.data_directory,
        "chains": f"{config.data_config.chain1}-{config.data_config.chain2}",
        "hyperparameters": "Single configuration for demo"
    })
    
    try:
        # Initialize RE-Net adapter
        logger.log_phase_start("initialization")
        renet_adapter = RENetAdapter(config)
        logger.log_phase_completion("initialization", {"status": "success"})
        
        # Prepare dataset
        logger.log_phase_start("dataset_preparation")
        if not renet_adapter.prepare_dataset():
            raise RuntimeError("Dataset preparation failed")
        logger.log_phase_completion("dataset_preparation", {"status": "success"})
        
        # Run pretraining
        logger.log_phase_start("pretraining")
        pretrain_result = renet_adapter.run_pretraining(
            dropout=0.5,
            n_hidden=100,
            learning_rate=0.001,
            max_epochs=20,
            batch_size=128
        )
        
        if not pretrain_result.success:
            raise RuntimeError(f"Pretraining failed: {pretrain_result.error_message}")
        
        logger.log_metrics("pretraining", {
            "execution_time": pretrain_result.execution_time,
            "status": "success"
        })
        logger.log_phase_completion("pretraining", {"status": "success"})
        
        # Run training
        logger.log_phase_start("training")
        train_result = renet_adapter.run_training(
            dropout=0.5,
            n_hidden=100,
            learning_rate=0.001,
            max_epochs=10,
            batch_size=128
        )
        
        if not train_result.success:
            raise RuntimeError(f"Training failed: {train_result.error_message}")
        
        logger.log_metrics("training", {
            "execution_time": train_result.execution_time,
            "status": "success"
        })
        logger.log_phase_completion("training", {"status": "success"})
        
        # Run testing
        logger.log_phase_start("testing")
        run_id = renet_adapter.generate_run_id()
        test_result = renet_adapter.run_testing(
            n_hidden=100,
            run_id=run_id
        )
        
        if not test_result.success:
            raise RuntimeError(f"Testing failed: {test_result.error_message}")
        
        logger.log_metrics("testing", {
            "execution_time": test_result.execution_time,
            "output_file": test_result.output_file,
            "status": "success"
        })
        logger.log_phase_completion("testing", {"status": "success"})
        
        # Create metadata
        results_directory = config.get_results_directory(run_id)
        metadata_file = renet_adapter.create_metadata_file(
            hyperparameters={
                "dropout": 0.5,
                "n_hidden": 100,
                "learning_rate": 0.001,
                "train_epochs": 10,
                "batch_size": 128
            },
            run_id=run_id,
            results_directory=results_directory,
            pretraining_epochs=20
        )
        
        # Cleanup
        logger.log_phase_start("cleanup")
        renet_adapter.cleanup_temporary_files()
        logger.log_phase_completion("cleanup", {"status": "success"})
        
        # Success summary
        logger.logger.info("Experiment completed successfully!")
        logger.logger.info(f"Results saved to: {test_result.output_file}")
        logger.logger.info(f"Metadata saved to: {metadata_file}")
        
        return True
        
    except Exception as e:
        logger.log_error(f"Experiment failed: {e}", e)
        return False


def demonstrate_configuration_validation():
    """Demonstrate configuration validation features."""
    
    print("Demonstrating configuration validation...")
    
    try:
        # This will fail validation - invalid train ratio
        bad_config = DataConfig(
            data_directory="/nonexistent/path",
            replica="test",
            chain1="A",
            chain2="B",
            train_ratio=1.5,  # Invalid: must be < 1.0
            validation_ratio=0.2
        )
    except ValueError as e:
        print(f"✓ Caught invalid train ratio: {e}")
    
    try:
        # This will fail validation - negative learning rate
        bad_hyperparams = HyperparameterConfig(
            learning_rates=[-0.001]  # Invalid: must be positive
        )
    except ValueError as e:
        print(f"✓ Caught invalid learning rate: {e}")
    
    print("Configuration validation working correctly!")


if __name__ == "__main__":
    print("RNN-MD Refactored Architecture - Basic Usage Example")
    print("=" * 60)
    
    # Demonstrate configuration validation
    demonstrate_configuration_validation()
    print()
    
    # Check if data directory exists before running experiment
    data_dir = "/path/to/your/data"  # TODO: Update this to match the data_directory in create_example_config()
    if not Path(data_dir).exists():
        print(f"⚠️  Data directory {data_dir} does not exist.")
        print("Please update the data_directory in create_example_config() to run the full example.")
        print()
        print("The refactored architecture provides:")
        print("- Centralized configuration with validation")
        print("- Clean interfaces to RE-Net functionality")
        print("- Comprehensive error handling and logging")
        print("- Type safety throughout the codebase")
        print("- Backward compatibility with existing scripts")
    else:
        print("Running full experiment example...")
        success = run_simple_experiment()
        if success:
            print("✓ Example experiment completed successfully!")
        else:
            print("✗ Example experiment failed. Check logs for details.")
    
    print("\nTo use this architecture in your own code:")
    print("1. Import the necessary modules from src/")
    print("2. Create configuration objects with validation")
    print("3. Use the RENetAdapter for clean RE-Net interaction")
    print("4. Leverage structured logging for debugging")
    print("\nSee docs/REFACTORING_GUIDE.md for detailed documentation.") 
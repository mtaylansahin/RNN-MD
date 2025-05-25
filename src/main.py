"""Main entry point for RNN-MD experiments using refactored architecture."""

import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Any
import itertools

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core.config import ConfigManager, ExperimentConfig
from core.utils import setup_logging, get_logger, ExperimentLogger
from adapters.renet import RENetAdapter


def run_data_preprocessing(config: ExperimentConfig, logger: ExperimentLogger) -> bool:
    """Run data preprocessing using the format.py script.
    
    Args:
        config: Experiment configuration
        logger: Experiment logger
        
    Returns:
        True if preprocessing successful, False otherwise
    """
    logger.log_phase_start("data_preprocessing")
    
    try:
        # Run format.py with configured parameters
        format_args = [
            config.data_config.data_directory,
            config.data_config.interaction_type,
            config.data_config.replica,
            config.data_config.chain1,
            config.data_config.chain2,
            str(config.data_config.train_ratio),
            str(config.data_config.validation_ratio)
        ]
        
        result = subprocess.run(
            [sys.executable, "format.py"] + format_args,
            check=True,
            capture_output=True,
            text=True
        )
        
        logger.logger.info("Data preprocessing completed successfully")
        logger.log_phase_completion("data_preprocessing", {"status": "success"})
        return True
        
    except subprocess.CalledProcessError as e:
        error_msg = f"Data preprocessing failed: {e.stderr}"
        logger.log_error(error_msg, e)
        logger.log_phase_completion("data_preprocessing", {"status": "failed", "error": error_msg})
        return False
    except Exception as e:
        error_msg = f"Unexpected error during data preprocessing: {e}"
        logger.log_error(error_msg, e)
        logger.log_phase_completion("data_preprocessing", {"status": "failed", "error": error_msg})
        return False


def run_hyperparameter_sweep(
    config: ExperimentConfig,
    renet_adapter: RENetAdapter,
    logger: ExperimentLogger
) -> List[Dict[str, Any]]:
    """Run hyperparameter sweep for all configured parameter combinations.
    
    Args:
        config: Experiment configuration
        renet_adapter: RE-Net adapter instance
        logger: Experiment logger
        
    Returns:
        List of results for each hyperparameter combination
    """
    logger.log_phase_start("hyperparameter_sweep")
    
    hyperparams = config.hyperparameters
    results = []
    
    # Generate all combinations of hyperparameters
    param_combinations = list(itertools.product(
        hyperparams.dropout_rates,
        hyperparams.hidden_units,
        hyperparams.learning_rates,
        hyperparams.pretrain_epochs,
        hyperparams.batch_sizes,
        hyperparams.train_epochs
    ))
    
    total_combinations = len(param_combinations)
    logger.logger.info(f"Starting hyperparameter sweep with {total_combinations} combinations")
    
    for run_id, (dropout, n_hidden, learning_rate, pretrain_epochs, batch_size, train_epochs) in enumerate(param_combinations, 1):
        
        hyperparameters = {
            "dropout": dropout,
            "n_hidden": n_hidden,
            "learning_rate": learning_rate,
            "pretrain_epochs": pretrain_epochs,
            "batch_size": batch_size,
            "train_epochs": train_epochs
        }
        
        logger.log_hyperparameter_run(run_id, hyperparameters)
        
        # Generate unique run ID
        run_identifier = renet_adapter.generate_run_id()
        results_directory = config.get_results_directory(run_identifier)
        
        try:
            # Run pretraining
            pretrain_result = renet_adapter.run_pretraining(
                dropout=dropout,
                n_hidden=n_hidden,
                learning_rate=learning_rate,
                max_epochs=pretrain_epochs,
                batch_size=batch_size
            )
            
            if not pretrain_result.success:
                logger.log_error(f"Pretraining failed for run {run_id}: {pretrain_result.error_message}")
                results.append({
                    "run_id": run_id,
                    "run_identifier": run_identifier,
                    "hyperparameters": hyperparameters,
                    "status": "failed",
                    "phase": "pretraining",
                    "error": pretrain_result.error_message
                })
                continue
            
            logger.log_metrics("pretraining", {
                "execution_time": pretrain_result.execution_time
            })
            
            # Run training
            train_result = renet_adapter.run_training(
                dropout=dropout,
                n_hidden=n_hidden,
                learning_rate=learning_rate,
                max_epochs=train_epochs,
                batch_size=batch_size
            )
            
            if not train_result.success:
                logger.log_error(f"Training failed for run {run_id}: {train_result.error_message}")
                results.append({
                    "run_id": run_id,
                    "run_identifier": run_identifier,
                    "hyperparameters": hyperparameters,
                    "status": "failed",
                    "phase": "training",
                    "error": train_result.error_message
                })
                continue
            
            logger.log_metrics("training", {
                "execution_time": train_result.execution_time
            })
            
            # Run testing
            test_result = renet_adapter.run_testing(
                n_hidden=n_hidden,
                run_id=run_identifier
            )
            
            if not test_result.success:
                logger.log_error(f"Testing failed for run {run_id}: {test_result.error_message}")
                results.append({
                    "run_id": run_id,
                    "run_identifier": run_identifier,
                    "hyperparameters": hyperparameters,
                    "status": "failed",
                    "phase": "testing",
                    "error": test_result.error_message
                })
                continue
            
            logger.log_metrics("testing", {
                "execution_time": test_result.execution_time
            })
            
            # Create metadata file
            metadata_file = renet_adapter.create_metadata_file(
                hyperparameters=hyperparameters,
                run_id=run_identifier,
                results_directory=results_directory,
                pretraining_epochs=pretrain_epochs
            )
            
            # Record successful result
            results.append({
                "run_id": run_id,
                "run_identifier": run_identifier,
                "hyperparameters": hyperparameters,
                "status": "success",
                "output_file": test_result.output_file,
                "metadata_file": metadata_file,
                "execution_times": {
                    "pretraining": pretrain_result.execution_time,
                    "training": train_result.execution_time,
                    "testing": test_result.execution_time
                }
            })
            
            logger.logger.info(f"Completed run {run_id}/{total_combinations} successfully")
            
        except Exception as e:
            error_msg = f"Unexpected error in run {run_id}: {e}"
            logger.log_error(error_msg, e)
            results.append({
                "run_id": run_id,
                "run_identifier": run_identifier,
                "hyperparameters": hyperparameters,
                "status": "failed",
                "phase": "unexpected_error",
                "error": error_msg
            })
    
    # Log summary
    successful_runs = sum(1 for r in results if r["status"] == "success")
    failed_runs = len(results) - successful_runs
    
    summary = {
        "total_runs": len(results),
        "successful_runs": successful_runs,
        "failed_runs": failed_runs,
        "success_rate": f"{(successful_runs / len(results) * 100):.1f}%" if results else "0%"
    }
    
    logger.log_phase_completion("hyperparameter_sweep", summary)
    return results


def main():
    """Main entry point for RNN-MD experiments."""
    try:
        # Parse configuration
        config_manager = ConfigManager()
        config = config_manager.parse_configuration()
        
        # Setup logging
        setup_logging(
            log_level="INFO",
            log_file="logs/rnn_md.log",
            experiment_name=config.experiment_name
        )
        
        # Initialize experiment logger
        experiment_logger = ExperimentLogger(config.experiment_name)
        
        # Log experiment start
        config_dict = {
            "experiment_name": config.experiment_name,
            "data_directory": config.data_config.data_directory,
            "replica": config.data_config.replica,
            "chains": f"{config.data_config.chain1}-{config.data_config.chain2}",
            "train_ratio": config.data_config.train_ratio,
            "validation_ratio": config.data_config.validation_ratio,
            "gpu_device": config.gpu_device,
            "random_seed": config.random_seed
        }
        experiment_logger.log_experiment_start(config_dict)
        
        # Save configuration for reproducibility
        config_manager.save_configuration(
            config,
            f"configs/{config.experiment_name}_config.json"
        )
        
        # Initialize RE-Net adapter
        renet_adapter = RENetAdapter(config)
        
        # Step 1: Data preprocessing
        if not run_data_preprocessing(config, experiment_logger):
            experiment_logger.log_error("Data preprocessing failed, aborting experiment")
            return 1
        
        # Step 2: Dataset preparation for RE-Net
        experiment_logger.log_phase_start("dataset_preparation")
        if not renet_adapter.prepare_dataset():
            experiment_logger.log_error("Dataset preparation failed, aborting experiment")
            return 1
        experiment_logger.log_phase_completion("dataset_preparation", {"status": "success"})
        
        # Step 3: Hyperparameter sweep
        results = run_hyperparameter_sweep(config, renet_adapter, experiment_logger)
        
        # Step 4: Cleanup
        experiment_logger.log_phase_start("cleanup")
        renet_adapter.cleanup_temporary_files()
        experiment_logger.log_phase_completion("cleanup", {"status": "success"})
        
        # Final summary
        successful_results = [r for r in results if r["status"] == "success"]
        experiment_logger.logger.info(f"Experiment completed: {len(successful_results)}/{len(results)} runs successful")
        
        if successful_results:
            experiment_logger.logger.info("Successful runs:")
            for result in successful_results:
                experiment_logger.logger.info(f"  Run {result['run_id']} ({result['run_identifier']}): {result['output_file']}")
        
        return 0 if successful_results else 1
        
    except KeyboardInterrupt:
        logger = get_logger(__name__)
        logger.info("Experiment interrupted by user")
        return 130
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Unexpected error in main: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
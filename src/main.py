"""Main entry point for RNN-MD experiments using refactored architecture."""

import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Any
import itertools
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core.config import ConfigManager, ExperimentConfig
from core.utils import setup_logging, get_logger, ExperimentLogger
from adapters.renet import RENetAdapter
from analysis import ResultsManager, AnalysisConfig


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
            
            # Generate visualization analysis for this successful run
            visualization_success = run_visualization_analysis(config, results[-1], logger)
            if visualization_success:
                results[-1]["visualization_status"] = "success"
                results[-1]["analysis_directory"] = os.path.join(results_directory, "analysis")
            else:
                results[-1]["visualization_status"] = "failed"
            
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
    visualization_successes = sum(1 for r in results if r.get("visualization_status") == "success")
    
    summary = {
        "total_runs": len(results),
        "successful_runs": successful_runs,
        "failed_runs": failed_runs,
        "success_rate": f"{(successful_runs / len(results) * 100):.1f}%" if results else "0%",
        "visualizations_generated": visualization_successes,
        "visualization_success_rate": f"{(visualization_successes / successful_runs * 100):.1f}%" if successful_runs > 0 else "0%"
    }
    
    logger.log_phase_completion("hyperparameter_sweep", summary)
    return results


def run_visualization_analysis(
    experiment_config: ExperimentConfig,
    result: Dict[str, Any],
    logger: ExperimentLogger
) -> bool:
    """Run visualization analysis for a successful model run.
    
    Args:
        experiment_config: Experiment configuration
        result: Result dictionary from successful model run
        logger: Experiment logger
        
    Returns:
        True if visualization generation successful, False otherwise
    """
    logger.log_phase_start(f"visualization_analysis_run_{result['run_id']}")
    
    try:
        # Determine input directory (original data directory)
        input_directory = experiment_config.data_config.data_directory
        
        # Get results directory
        results_directory = experiment_config.get_results_directory(result['run_identifier'])
        
        # Create analysis subdirectory within results
        analysis_output_dir = os.path.join(results_directory, "analysis")
        
        # Get the prediction output file
        prediction_file = result['output_file']
        
        # Create analysis configuration
        analysis_config = AnalysisConfig(
            input_directory=input_directory,
            output_directory=analysis_output_dir,
            output_file_path=prediction_file,
            num_pairs_to_show=50,  # Can be made configurable
            valid_steps_to_show=20  # Can be made configurable
        )
        
        # Run analysis and visualization
        results_manager = ResultsManager(analysis_config)
        success = results_manager.run_complete_analysis()
        
        if success:
            logger.logger.info(f"Visualization analysis completed for run {result['run_id']}")
            logger.log_phase_completion(
                f"visualization_analysis_run_{result['run_id']}", 
                {
                    "status": "success",
                    "analysis_directory": analysis_output_dir,
                    "prediction_file": prediction_file
                }
            )
            return True
        else:
            logger.log_error(f"Visualization analysis failed for run {result['run_id']}")
            logger.log_phase_completion(
                f"visualization_analysis_run_{result['run_id']}", 
                {"status": "failed", "error": "Analysis pipeline failed"}
            )
            return False
        
    except Exception as e:
        error_msg = f"Visualization analysis failed for run {result['run_id']}: {e}"
        logger.log_error(error_msg, e)
        logger.log_phase_completion(
            f"visualization_analysis_run_{result['run_id']}", 
            {"status": "failed", "error": error_msg}
        )
        return False


def generate_experiment_summary_visualizations(
    config: ExperimentConfig,
    results: List[Dict[str, Any]],
    logger: ExperimentLogger
) -> bool:
    """Generate summary visualizations across all successful runs.
    
    Args:
        config: Experiment configuration
        results: List of all run results
        logger: Experiment logger
        
    Returns:
        True if summary visualization generation successful, False otherwise
    """
    logger.log_phase_start("experiment_summary_visualizations")
    
    try:
        successful_results = [r for r in results if r["status"] == "success" and r.get("visualization_status") == "success"]
        
        if not successful_results:
            logger.logger.warning("No successful runs with visualizations found for summary generation")
            logger.log_phase_completion("experiment_summary_visualizations", {"status": "skipped", "reason": "no_successful_runs"})
            return False
        
        # Create experiment-wide summary directory
        summary_directory = os.path.join(config.results_base_directory, f"{config.experiment_name}_experiment_summary")
        Path(summary_directory).mkdir(parents=True, exist_ok=True)
        
        logger.logger.info(f"Generated experiment summary for {len(successful_results)} successful runs")
        logger.logger.info(f"Summary directory: {summary_directory}")
        logger.logger.info("Individual run visualizations available at:")
        
        for result in successful_results:
            if "analysis_directory" in result:
                logger.logger.info(f"  Run {result['run_id']} ({result['run_identifier']}): {result['analysis_directory']}")
        
        logger.log_phase_completion(
            "experiment_summary_visualizations", 
            {
                "status": "success",
                "summary_directory": summary_directory,
                "successful_runs_with_viz": len(successful_results)
            }
        )
        return True
        
    except Exception as e:
        error_msg = f"Experiment summary visualization generation failed: {e}"
        logger.log_error(error_msg, e)
        logger.log_phase_completion("experiment_summary_visualizations", {"status": "failed", "error": error_msg})
        return False


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
        
        # Generate experiment summary visualizations
        if generate_experiment_summary_visualizations(config, results, experiment_logger):
            experiment_logger.logger.info("Experiment summary visualizations generated successfully")
        else:
            experiment_logger.logger.warning("Experiment summary visualizations generation skipped")
        
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
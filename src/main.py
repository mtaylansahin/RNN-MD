"""Main entry point for RNN-MD experiments using refactored architecture."""

import sys
from pathlib import Path
import os
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core.config import ConfigManager, ExperimentConfig
from core.utils import setup_logging, get_logger, ExperimentLogger
from adapters.renet import RENetAdapter
from analysis import ResultsManager, AnalysisConfig
from pipelines.data_preprocessor import run_preprocessing


def run_data_preprocessing(config: ExperimentConfig, logger: ExperimentLogger, run_id: str) -> str:
    """Run data preprocessing using in-process preprocessor and return output directory."""
    logger.log_phase_start("data_preprocessing")
    try:
        preprocess_dir = os.path.join(
            config.get_results_directory(run_id),
            "preprocess"
        )
        Path(preprocess_dir).mkdir(parents=True, exist_ok=True)
        run_preprocessing(
            data_directory=config.data_config.data_directory,
            interaction_type=config.data_config.interaction_type,
            replica=config.data_config.replica,
            chain1=config.data_config.chain1,
            chain2=config.data_config.chain2,
            train_ratio=config.data_config.train_ratio,
            validation_ratio=config.data_config.validation_ratio,
            output_directory=preprocess_dir
        )
        logger.logger.info("Data preprocessing completed successfully")
        logger.log_phase_completion("data_preprocessing", {"status": "success", "output_directory": preprocess_dir})
        return preprocess_dir
    except Exception as e:
        error_msg = f"Unexpected error during data preprocessing: {e}"
        logger.log_error(error_msg, e)
        logger.log_phase_completion("data_preprocessing", {"status": "failed", "error": error_msg})
        raise


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
        results_directory = experiment_config.get_results_directory(result['run_identifier'])
        input_directory = result.get('preprocess_directory', experiment_config.data_config.data_directory)

        analysis_output_dir = os.path.join(results_directory, "analysis")

        prediction_file = result['output_file']

        analysis_config = AnalysisConfig(
            input_directory=input_directory,
            output_directory=analysis_output_dir,
            output_file_path=prediction_file
        )

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


def main():
    """Main entry point for RNN-MD experiments."""
    try:
        config_manager = ConfigManager()
        config = config_manager.parse_configuration()

        setup_logging(
            log_level="INFO",
            log_file="logs/rnn_md.log",
            experiment_name=config.experiment_name
        )

        experiment_logger = ExperimentLogger(config.experiment_name)

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

        renet_adapter = RENetAdapter(config)

        run_id = renet_adapter.generate_run_id()
        dataset_name = f"{config.experiment_name}_{run_id}"

        preprocess_dir = run_data_preprocessing(config, experiment_logger, run_id)

        experiment_logger.log_phase_start("dataset_preparation")
        if not renet_adapter.prepare_dataset(preprocess_directory=preprocess_dir, dataset_name=dataset_name):
            experiment_logger.log_error("Dataset preparation failed, aborting experiment")
            return 1
        experiment_logger.log_phase_completion("dataset_preparation", {"status": "success"})

        hp = config.hyperparameters
        dropout = hp.dropout_rates[0]
        n_hidden = hp.hidden_units[0]
        learning_rate = hp.learning_rates[0]
        pretrain_epochs = hp.pretrain_epochs[0]
        train_epochs = hp.train_epochs[0]
        batch_size = hp.batch_sizes[0]

        experiment_logger.log_phase_start("training_pipeline", {
            "dropout": dropout,
            "n_hidden": n_hidden,
            "learning_rate": learning_rate,
            "pretrain_epochs": pretrain_epochs,
            "train_epochs": train_epochs,
            "batch_size": batch_size
        })
        train_result = renet_adapter.train(
            dataset_name=dataset_name,
            dropout=dropout,
            n_hidden=n_hidden,
            learning_rate=learning_rate,
            pretrain_epochs=pretrain_epochs,
            train_epochs=train_epochs,
            batch_size=batch_size
        )
        if not train_result.success:
            experiment_logger.log_error(f"Training pipeline failed: {train_result.error_message}")
            return 1
        experiment_logger.log_phase_completion("training_pipeline", {"execution_time": train_result.execution_time})

        results_directory = config.get_results_directory(run_id)
        outputs_directory = os.path.join(results_directory, "outputs")
        experiment_logger.log_phase_start("testing", {"n_hidden": n_hidden})
        test_result = renet_adapter.run_testing(
            dataset_name=dataset_name,
            n_hidden=n_hidden,
            run_id=run_id,
            results_directory=outputs_directory
        )
        if not test_result.success:
            experiment_logger.log_error(f"Testing failed: {test_result.error_message}")
            return 1
        experiment_logger.log_phase_completion("testing", {"execution_time": test_result.execution_time,
                                                           "output_file": test_result.output_file})

        _ = renet_adapter.create_metadata_file(
            hyperparameters={
                "dropout": dropout,
                "n_hidden": n_hidden,
                "learning_rate": learning_rate,
                "train_epochs": train_epochs,
                "batch_size": batch_size
            },
            run_id=run_id,
            results_directory=results_directory,
            pretraining_epochs=pretrain_epochs
        )

        result_record = {
            "run_id": 1,
            "run_identifier": run_id,
            "status": "success",
            "output_file": test_result.output_file,
            "dataset_name": dataset_name,
            "preprocess_directory": preprocess_dir,
            "results_directory": results_directory
        }
        _ = run_visualization_analysis(config, result_record, experiment_logger)

        experiment_logger.log_phase_start("cleanup")
        renet_adapter.cleanup_temporary_files()
        experiment_logger.log_phase_completion("cleanup", {"status": "success"})

        experiment_logger.logger.info("Experiment completed successfully")
        return 0

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

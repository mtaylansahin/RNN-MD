"""Backward-compatible wrapper for RNN-MD experiments.

This script maintains the original command-line interface while using
the refactored architecture internally.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for refactored imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from core.config import ConfigManager, DataConfig, HyperparameterConfig, ExperimentConfig
from core.utils import setup_logging, get_logger


def parse_original_arguments():
    """Parse command line arguments in the original format."""
    parser = argparse.ArgumentParser(
        description='RNN-MD: Protein-protein interaction dynamics prediction (Refactored)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Original arguments from RNN-MD.py
    parser.add_argument('--data_dir', type=str, required=True, 
                       help='Directory containing the interaction files.')
    parser.add_argument('--replica', type=str, required=True, 
                       help='Replica number.')
    parser.add_argument('--chain1', type=str, required=True, 
                       help='First chain.')
    parser.add_argument('--chain2', type=str, required=True, 
                       help='Second chain.')
    parser.add_argument('--train_ratio', type=float, required=True, 
                       help='Training ratio for splitting the data.')
    parser.add_argument('--valid_ratio', type=float, required=True, 
                       help='Validation ratio for splitting the data.')
    parser.add_argument('--dropout', type=str, default="0.5", 
                       help='Dropout rate for training.')
    parser.add_argument('--learning_rate', type=str, default="0.001", 
                       help='Learning rate for the optimizer.')
    parser.add_argument('--batch_size', type=str, default="128", 
                       help='Batch size for training.')
    parser.add_argument('--pretrain_epochs', type=str, default="30", 
                       help='Number of epochs for pre-training.')
    parser.add_argument('--train_epochs', type=str, default="10", 
                       help='Number of epochs for training.')
    parser.add_argument('--n_hidden', default="100", type=str, 
                       help='Number of hidden units in the hidden layer.')
    
    # Additional arguments for system configuration
    parser.add_argument('--gpu', type=int, default=0, 
                       help='GPU device to use')
    parser.add_argument('--interaction_type', type=str, default='residue',
                       choices=['residue', 'atomic'],
                       help='Type of interaction analysis')
    parser.add_argument('--seed', type=int, default=999,
                       help='Random seed for reproducibility')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')

    return parser.parse_args()


def convert_to_new_config(args) -> ExperimentConfig:
    """Convert original arguments to new configuration format.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        ExperimentConfig instance
    """
    # Parse parameter ranges using the original logic
    config_manager = ConfigManager()
    
    # Create data configuration
    data_config = DataConfig(
        data_directory=args.data_dir,
        replica=args.replica,
        chain1=args.chain1,
        chain2=args.chain2,
        train_ratio=args.train_ratio,
        validation_ratio=args.valid_ratio,
        interaction_type=args.interaction_type
    )
    
    # Parse hyperparameters
    hyperparameters = HyperparameterConfig(
        dropout_rates=config_manager._parse_parameter_range(args.dropout, float),
        learning_rates=config_manager._parse_parameter_range(args.learning_rate, float),
        batch_sizes=config_manager._parse_parameter_range(args.batch_size, int),
        pretrain_epochs=config_manager._parse_parameter_range(args.pretrain_epochs, int),
        train_epochs=config_manager._parse_parameter_range(args.train_epochs, int),
        hidden_units=config_manager._parse_parameter_range(args.n_hidden, int)
    )
    
    # Create experiment configuration
    experiment_name = Path(args.data_dir).name
    config = ExperimentConfig(
        experiment_name=experiment_name,
        data_config=data_config,
        hyperparameters=hyperparameters,
        gpu_device=args.gpu,
        random_seed=args.seed
    )
    
    return config


def main():
    """Main entry point that maintains backward compatibility."""
    try:
        # Parse original arguments
        args = parse_original_arguments()
        
        # Convert to new configuration format
        config = convert_to_new_config(args)
        
        # Setup logging
        setup_logging(
            log_level=args.log_level,
            log_file=f"logs/{config.experiment_name}.log",
            experiment_name=config.experiment_name
        )
        
        logger = get_logger(__name__)
        logger.info("Starting RNN-MD experiment with refactored architecture")
        logger.info(f"Experiment: {config.experiment_name}")
        logger.info(f"Data directory: {config.data_config.data_directory}")
        logger.info(f"Chains: {config.data_config.chain1}-{config.data_config.chain2}")
        
        # Import and run the new main function
        from main import main as new_main
        
        # Temporarily override sys.argv to pass config to new main
        original_argv = sys.argv
        try:
            # Create argument list that the new ConfigManager can parse
            new_args = [
                '--data_dir', config.data_config.data_directory,
                '--replica', config.data_config.replica,
                '--chain1', config.data_config.chain1,
                '--chain2', config.data_config.chain2,
                '--train_ratio', str(config.data_config.train_ratio),
                '--valid_ratio', str(config.data_config.validation_ratio),
                '--dropout', args.dropout,
                '--learning_rate', args.learning_rate,
                '--batch_size', args.batch_size,
                '--pretrain_epochs', args.pretrain_epochs,
                '--train_epochs', args.train_epochs,
                '--n_hidden', args.n_hidden,
                '--gpu', str(config.gpu_device),
                '--seed', str(config.random_seed),
                '--interaction_type', config.data_config.interaction_type
            ]
            
            sys.argv = ['main.py'] + new_args
            
            # Run the new main function
            return new_main()
            
        finally:
            sys.argv = original_argv
        
    except KeyboardInterrupt:
        logger = get_logger(__name__)
        logger.info("Experiment interrupted by user")
        return 130
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Error in backward-compatible wrapper: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 
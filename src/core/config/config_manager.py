"""Configuration manager for parsing and validating experiment configurations."""

import argparse
import json
import os
from typing import List, Union, Optional
import numpy as np

from .experiment_config import ExperimentConfig, HyperparameterConfig, DataConfig


class ConfigManager:
    """Manages configuration parsing, validation, and loading for experiments."""
    
    def __init__(self):
        """Initialize the configuration manager."""
        self._parser = self._create_argument_parser()
    
    def _create_argument_parser(self) -> argparse.ArgumentParser:
        """Create and configure the argument parser."""
        parser = argparse.ArgumentParser(
            description='RNN-MD: Protein-protein interaction dynamics prediction',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        # Data configuration arguments
        data_group = parser.add_argument_group('Data Configuration')
        data_group.add_argument(
            '--data_dir', 
            type=str, 
            required=True,
            help='Directory containing the interaction files'
        )
        data_group.add_argument(
            '--replica', 
            type=str, 
            required=True,
            help='Replica number for the experiment'
        )
        data_group.add_argument(
            '--chain1', 
            type=str, 
            required=True,
            help='First chain identifier'
        )
        data_group.add_argument(
            '--chain2', 
            type=str, 
            required=True,
            help='Second chain identifier'
        )
        data_group.add_argument(
            '--train_ratio', 
            type=float, 
            required=True,
            help='Training ratio for splitting the data (0.0-1.0)'
        )
        data_group.add_argument(
            '--valid_ratio', 
            type=float, 
            required=True,
            help='Validation ratio for splitting the data (0.0-1.0)'
        )
        data_group.add_argument(
            '--interaction_type',
            type=str,
            default='residue',
            choices=['residue', 'atomic'],
            help='Type of interaction analysis'
        )
        
        # Hyperparameter configuration arguments
        hyperparam_group = parser.add_argument_group('Hyperparameter Configuration')
        hyperparam_group.add_argument(
            '--dropout',
            type=str,
            default='0.5',
            help='Dropout rate(s). Single value or range [start,stop,step]'
        )
        hyperparam_group.add_argument(
            '--learning_rate',
            type=str,
            default='0.001',
            help='Learning rate(s). Single value or range [start,stop,step]'
        )
        hyperparam_group.add_argument(
            '--batch_size',
            type=str,
            default='128',
            help='Batch size(s). Single value or range [start,stop,step]'
        )
        hyperparam_group.add_argument(
            '--pretrain_epochs',
            type=str,
            default='30',
            help='Pretraining epoch(s). Single value or range [start,stop,step]'
        )
        hyperparam_group.add_argument(
            '--train_epochs',
            type=str,
            default='10',
            help='Training epoch(s). Single value or range [start,stop,step]'
        )
        hyperparam_group.add_argument(
            '--n_hidden',
            type=str,
            default='100',
            help='Hidden unit(s). Single value or range [start,stop,step]'
        )
        
        # System configuration arguments
        system_group = parser.add_argument_group('System Configuration')
        system_group.add_argument(
            '--gpu',
            type=int,
            default=0,
            help='GPU device ID to use'
        )
        system_group.add_argument(
            '--seed',
            type=int,
            default=999,
            help='Random seed for reproducibility'
        )
        system_group.add_argument(
            '--config_file',
            type=str,
            help='Path to JSON configuration file (overrides command line args)'
        )
        
        return parser
    
    def parse_configuration(self, args: Optional[List[str]] = None) -> ExperimentConfig:
        """Parse command line arguments or config file into ExperimentConfig.
        
        Args:
            args: Optional list of arguments to parse. If None, uses sys.argv
            
        Returns:
            Validated ExperimentConfig instance
            
        Raises:
            ValueError: If configuration validation fails
            FileNotFoundError: If config file is specified but not found
        """
        parsed_args = self._parser.parse_args(args)
        
        # Load from config file if specified
        if parsed_args.config_file:
            return self._load_from_config_file(parsed_args.config_file)
        
        # Parse hyperparameters from command line
        hyperparameters = self._parse_hyperparameters(parsed_args)
        
        # Create data configuration
        data_config = DataConfig(
            data_directory=parsed_args.data_dir,
            replica=parsed_args.replica,
            chain1=parsed_args.chain1,
            chain2=parsed_args.chain2,
            train_ratio=parsed_args.train_ratio,
            validation_ratio=parsed_args.valid_ratio,
            interaction_type=parsed_args.interaction_type
        )
        
        # Create experiment configuration
        experiment_name = os.path.basename(parsed_args.data_dir.rstrip('/'))
        experiment_config = ExperimentConfig(
            experiment_name=experiment_name,
            data_config=data_config,
            hyperparameters=hyperparameters,
            gpu_device=parsed_args.gpu,
            random_seed=parsed_args.seed
        )
        
        return experiment_config
    
    def _parse_hyperparameters(self, args: argparse.Namespace) -> HyperparameterConfig:
        """Parse hyperparameter arguments into HyperparameterConfig.
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            Validated HyperparameterConfig instance
        """
        return HyperparameterConfig(
            dropout_rates=self._parse_parameter_range(args.dropout, float),
            learning_rates=self._parse_parameter_range(args.learning_rate, float),
            batch_sizes=self._parse_parameter_range(args.batch_size, int),
            pretrain_epochs=self._parse_parameter_range(args.pretrain_epochs, int),
            train_epochs=self._parse_parameter_range(args.train_epochs, int),
            hidden_units=self._parse_parameter_range(args.n_hidden, int)
        )
    
    def _parse_parameter_range(self, param_str: str, param_type: type) -> List[Union[int, float]]:
        """Parse parameter string into list of values.
        
        Supports single values or ranges in format [start,stop,step].
        
        Args:
            param_str: Parameter string to parse
            param_type: Type to convert values to (int or float)
            
        Returns:
            List of parameter values
            
        Raises:
            ValueError: If parameter format is invalid
        """
        if param_str is None:
            return []
        
        try:
            if param_str.startswith('[') and param_str.endswith(']'):
                # Parse range format [start,stop,step] or [val1,val2,val3,...]
                values_str = param_str[1:-1].split(',')
                values = [param_type(val.strip()) for val in values_str]
                
                if len(values) == 3 and param_type == float:
                    # Range format for floats
                    start, stop, step = values
                    return list(np.arange(start, stop, step))
                elif len(values) == 3 and param_type == int:
                    # Range format for ints
                    start, stop, step = values
                    return list(range(start, stop, step))
                else:
                    # List of explicit values
                    return values
            else:
                # Single value
                return [param_type(param_str)]
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid parameter format '{param_str}': {e}")
    
    def _load_from_config_file(self, config_file_path: str) -> ExperimentConfig:
        """Load configuration from JSON file.
        
        Args:
            config_file_path: Path to the JSON configuration file
            
        Returns:
            Loaded ExperimentConfig instance
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file format is invalid
        """
        if not os.path.exists(config_file_path):
            raise FileNotFoundError(f"Configuration file not found: {config_file_path}")
        
        try:
            with open(config_file_path, 'r') as f:
                config_data = json.load(f)
            
            # Parse data configuration
            data_config = DataConfig(**config_data['data'])
            
            # Parse hyperparameter configuration
            hyperparameters = HyperparameterConfig(**config_data['hyperparameters'])
            
            # Parse experiment configuration
            experiment_config = ExperimentConfig(
                data_config=data_config,
                hyperparameters=hyperparameters,
                **config_data.get('experiment', {})
            )
            
            return experiment_config
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise ValueError(f"Invalid configuration file format: {e}")
    
    def save_configuration(self, config: ExperimentConfig, output_path: str) -> None:
        """Save configuration to JSON file.
        
        Args:
            config: ExperimentConfig to save
            output_path: Path where to save the configuration
        """
        config_dict = {
            'experiment': {
                'experiment_name': config.experiment_name,
                'gpu_device': config.gpu_device,
                'use_cuda': config.use_cuda,
                'random_seed': config.random_seed,
                'renet_directory': config.renet_directory,
                'results_base_directory': config.results_base_directory,
                'models_directory': config.models_directory,
                'sequence_length': config.sequence_length,
                'num_k_parameter': config.num_k_parameter,
                'model_type': config.model_type,
                'maxpool': config.maxpool,
                'gradient_norm_clip': config.gradient_norm_clip,
                'weight_decay': config.weight_decay,
                'validation_frequency': config.validation_frequency
            },
            'data': {
                'data_directory': config.data_config.data_directory,
                'replica': config.data_config.replica,
                'chain1': config.data_config.chain1,
                'chain2': config.data_config.chain2,
                'train_ratio': config.data_config.train_ratio,
                'validation_ratio': config.data_config.validation_ratio,
                'interaction_type': config.data_config.interaction_type
            },
            'hyperparameters': {
                'dropout_rates': config.hyperparameters.dropout_rates,
                'learning_rates': config.hyperparameters.learning_rates,
                'batch_sizes': config.hyperparameters.batch_sizes,
                'pretrain_epochs': config.hyperparameters.pretrain_epochs,
                'train_epochs': config.hyperparameters.train_epochs,
                'hidden_units': config.hyperparameters.hidden_units
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2) 
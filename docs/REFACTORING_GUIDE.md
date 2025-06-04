# RNN-MD Refactoring Guide

## Overview

This document describes the comprehensive refactoring of the RNN-MD project, which improves code organization, maintainability, error handling, and provides better separation of concerns while maintaining full backward compatibility.

## Key Improvements

### 1. **Clean Architecture & Separation of Concerns**
- **Configuration Management**: Centralized, validated configuration system
- **Adapter Pattern**: Clean interface to RE-Net without tight coupling
- **Service Layer**: Business logic separated from infrastructure
- **Utilities**: Reusable components for file management, logging, and process execution

### 2. **Robust Error Handling**
- Comprehensive exception handling with specific error types
- Proper process management replacing dangerous `os.system()` calls
- Graceful error recovery and rollback mechanisms
- Structured logging for debugging and monitoring

### 3. **Type Safety & Validation**
- Full type hints throughout the codebase
- Runtime validation of configuration parameters
- Data classes for structured information exchange
- Clear interfaces between components

### 4. **Maintainability & Testing**
- Modular design with single responsibility principle
- Dependency injection for easier testing
- Clear separation between pure logic and I/O operations
- Comprehensive logging for troubleshooting

## Architecture Overview

```
src/
├── core/
│   ├── config/           # Configuration management
│   │   ├── config_manager.py
│   │   ├── experiment_config.py
│   │   └── __init__.py
│   └── utils/            # Shared utilities
│       ├── file_utils.py
│       ├── logging_utils.py
│       ├── process_utils.py
│       └── __init__.py
├── adapters/
│   └── renet/            # RE-Net adapter layer
│       ├── renet_adapter.py
│       ├── model_manager.py
│       └── __init__.py
└── main.py              # New main entry point
```

## Usage Guide

### 1. Using the New Architecture

The new main entry point provides the same functionality with improved reliability:

```bash
python src/main.py \
    --data_dir /path/to/data \
    --replica replica1 \
    --chain1 A \
    --chain2 B \
    --train_ratio 0.7 \
    --valid_ratio 0.2 \
    --dropout "[0.3,0.5,0.7]" \
    --learning_rate "0.001" \
    --batch_size "128" \
    --pretrain_epochs "30" \
    --train_epochs "10" \
    --n_hidden "100"
```

### 2. Backward Compatibility

The original interface is preserved through a compatibility wrapper:

```bash
python RNN-MD-refactored.py \
    --data_dir /path/to/data \
    --replica replica1 \
    --chain1 A \
    --chain2 B \
    --train_ratio 0.7 \
    --valid_ratio 0.2 \
    --dropout "0.5" \
    --learning_rate "0.001" \
    --batch_size "128" \
    --pretrain_epochs "30" \
    --train_epochs "10" \
    --n_hidden "100"
```

### 3. Configuration Files

You can now use JSON configuration files for complex experiments:

```json
{
  "experiment": {
    "experiment_name": "protein_dynamics_exp1",
    "gpu_device": 0,
    "random_seed": 999
  },
  "data": {
    "data_directory": "/path/to/data",
    "replica": "replica1",
    "chain1": "A",
    "chain2": "B",
    "train_ratio": 0.7,
    "validation_ratio": 0.2,
    "interaction_type": "residue"
  },
  "hyperparameters": {
    "dropout_rates": [0.3, 0.5, 0.7],
    "learning_rates": [0.001, 0.01],
    "batch_sizes": [64, 128],
    "pretrain_epochs": [20, 30],
    "train_epochs": [10, 15],
    "hidden_units": [100, 200]
  }
}
```

Use with:
```bash
python src/main.py --config_file config.json
```

## Key Components

### Configuration Management

The new configuration system provides:
- **Validation**: All parameters are validated at startup
- **Type Safety**: Proper type conversion and checking
- **Documentation**: Self-documenting configuration classes
- **Flexibility**: Support for both command-line and file-based configuration

```python
from src.core.config import ConfigManager

config_manager = ConfigManager()
config = config_manager.parse_configuration()
# All configuration is now validated and type-safe
```

### RE-Net Adapter

Clean interface to RE-Net functionality:
- **Process Management**: Safe subprocess execution
- **File Management**: Organized file operations
- **Error Handling**: Comprehensive error recovery
- **Logging**: Structured logging throughout

```python
from src.adapters.renet import RENetAdapter

adapter = RENetAdapter(config)
result = adapter.run_pretraining(
    dropout=0.5,
    n_hidden=100,
    learning_rate=0.001,
    max_epochs=30,
    batch_size=128
)
```

### Utilities

Reusable utilities for common operations:
- **FileManager**: Safe file operations with error handling
- **ProcessManager**: Subprocess execution with timeouts and logging
- **ExperimentLogger**: Structured experiment tracking

## Migration Guide

### For Existing Scripts

1. **No Changes Required**: Existing scripts continue to work with `RNN-MD-refactored.py`
2. **Gradual Migration**: Can gradually adopt new features like configuration files
3. **Enhanced Logging**: Automatically get better error messages and logging

### For Development

1. **Use Type Hints**: All new code should include proper type annotations
2. **Configuration**: Use the centralized configuration system
3. **Error Handling**: Use specific exception types and proper error recovery
4. **Logging**: Use the structured logging system

## Testing

The refactored architecture enables better testing:

```python
# Example test structure
def test_configuration_validation():
    config = ExperimentConfig(...)
    # Configuration validation happens automatically
    
def test_renet_adapter():
    mock_config = create_mock_config()
    adapter = RENetAdapter(mock_config)
    # Test with mocked dependencies
```

## Performance Improvements

1. **Process Management**: Proper subprocess handling prevents resource leaks
2. **File Operations**: Efficient file management with validation
3. **Error Recovery**: Faster failure detection and recovery
4. **Logging**: Configurable logging levels for production vs debugging

## Monitoring & Debugging

### Structured Logging

All operations are logged with structured information:
```
2024-01-15 10:30:00 - experiment.protein_exp1 - INFO - Starting pretraining with parameters: {'dropout': 0.5, 'n_hidden': 100}
2024-01-15 10:30:15 - process.renet - INFO - Command completed successfully in 15.23s
2024-01-15 10:30:15 - experiment.protein_exp1 - INFO - Pretraining completed successfully
```

### Error Tracking

Comprehensive error information with context:
```
2024-01-15 10:35:00 - process.renet - ERROR - Command failed with return code 1: CUDA out of memory
2024-01-15 10:35:00 - experiment.protein_exp1 - ERROR - Pretraining failed for run 3: CUDA out of memory
```

### Configuration Tracking

All experiment configurations are automatically saved for reproducibility:
```
configs/
└── protein_exp1_config.json  # Complete configuration for experiment
```

## Benefits Summary

1. **Reliability**: Robust error handling and process management
2. **Maintainability**: Clean code structure following best practices
3. **Extensibility**: Easy to add new features and adapt to changes
4. **Debuggability**: Comprehensive logging and error reporting
5. **Type Safety**: Full type checking for better development experience
6. **Compatibility**: Existing scripts continue to work unchanged
7. **Performance**: Better resource management and error recovery
8. **Documentation**: Self-documenting code with clear interfaces

## Future Enhancements

The refactored architecture enables:
1. **Web Interface**: Easy to add REST API or web dashboard
2. **Distributed Computing**: Clean interfaces for cluster deployment
3. **Model Versioning**: Enhanced model management capabilities
4. **Real-time Monitoring**: Integration with monitoring systems
5. **Configuration Management**: Advanced configuration templating
6. **Testing Framework**: Comprehensive test suite development 
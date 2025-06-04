#!/usr/bin/env python3
"""Refactored results analysis script using the new RNN-MD architecture.

This script maintains backward compatibility with the original result.py
while using the new clean architecture internally.
"""

import sys
import argparse
from pathlib import Path

# Add src to path for refactored imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from analysis import ResultsManager, AnalysisConfig
from core.utils import setup_logging, get_logger


def parse_original_arguments():
    """Parse command line arguments in the original format."""
    parser = argparse.ArgumentParser(
        description="Process RNN-MD results and generate analysis (Refactored)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Original arguments from result.py
    parser.add_argument('--input_dir', type=str, required=True, 
                       help='Directory containing train.txt, test.txt, and valid.txt files')
    parser.add_argument('--output_dir', type=str, required=True, 
                       help='Directory to store the output files')
    parser.add_argument('--output_file_dir', type=str, required=True, 
                       help='Path to the prediction output file')
    
    # Additional parameters for enhanced functionality
    parser.add_argument('--num_pairs_to_show', type=int, default=50,
                       help='Number of pairs to show in heatmap visualizations')
    parser.add_argument('--valid_steps_to_show', type=int, default=20,
                       help='Number of validation steps to show in trajectories')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    return parser.parse_args()


def main():
    """Main entry point maintaining backward compatibility."""
    try:
        # Parse arguments using original format
        args = parse_original_arguments()
        
        # Setup logging
        experiment_name = Path(args.input_dir).name
        setup_logging(
            log_level=args.log_level,
            log_file=f"logs/analysis_{experiment_name}.log",
            experiment_name=f"analysis_{experiment_name}"
        )
        
        logger = get_logger(__name__)
        logger.info("Starting RNN-MD results analysis with refactored architecture")
        logger.info(f"Input directory: {args.input_dir}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Prediction file: {args.output_file_dir}")
        
        # Create configuration using new architecture
        config = AnalysisConfig(
            input_directory=args.input_dir,
            output_directory=args.output_dir,
            output_file_path=args.output_file_dir,
            num_pairs_to_show=args.num_pairs_to_show,
            valid_steps_to_show=args.valid_steps_to_show
        )
        
        # Run analysis using new architecture
        results_manager = ResultsManager(config)
        success = results_manager.run_complete_analysis()
        
        if success:
            logger.info("Analysis completed successfully")
            logger.info("Generated files:")
            logger.info(f"  - Performance metrics: {args.output_dir}/PerformanceMetrics.txt")
            logger.info(f"  - Ground truth data: {args.output_dir}/ground_truth.json")
            logger.info(f"  - Prediction data: {args.output_dir}/prediction.json")
            logger.info(f"  - Similarity score: {args.output_dir}/heatmap_similarity_score.txt")
            return 0
        else:
            logger.error("Analysis failed")
            return 1
        
    except KeyboardInterrupt:
        logger = get_logger(__name__)
        logger.info("Analysis interrupted by user")
        return 130
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Error in refactored results analysis: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 
"""Main results manager that orchestrates the complete analysis pipeline."""

import argparse
import os
from pathlib import Path
from typing import Optional, Dict, Any

from .config import AnalysisConfig
from .data import DataLoader, DataProcessor, LoadedData, ProcessedData
from .analytics import MetricsCalculator, MetricsReport
from ..core.utils import get_logger, FileManager, ExperimentLogger


logger = get_logger(__name__)


class ResultsManager:
    """Main manager for the complete results analysis pipeline."""
    
    def __init__(self, config: AnalysisConfig):
        """Initialize results manager.
        
        Args:
            config: Analysis configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.file_manager = FileManager()
        self.data_loader = DataLoader()
        self.data_processor = DataProcessor()
        self.metrics_calculator = MetricsCalculator()
        
        # Initialize experiment logger
        experiment_name = Path(config.input_directory).name
        self.experiment_logger = ExperimentLogger(f"analysis_{experiment_name}")
    
    def run_complete_analysis(self) -> bool:
        """Run the complete analysis pipeline.
        
        Returns:
            True if analysis completed successfully, False otherwise
        """
        self.logger.info("Starting complete results analysis")
        
        try:
            # Log experiment start
            config_dict = {
                "input_directory": self.config.input_directory,
                "output_directory": self.config.output_directory,
                "output_file": self.config.output_file_path,
                "num_pairs_to_show": self.config.num_pairs_to_show,
                "valid_steps_to_show": self.config.valid_steps_to_show
            }
            self.experiment_logger.log_experiment_start(config_dict)
            
            # Phase 1: Validation
            if not self._validate_inputs():
                return False
            
            # Phase 2: Data Loading
            loaded_data = self._load_data()
            if loaded_data is None:
                return False
            
            # Phase 3: Data Processing
            processed_data = self._process_data(loaded_data)
            if processed_data is None:
                return False
            
            # Phase 4: Metrics Calculation
            metrics_report = self._calculate_metrics(processed_data)
            if metrics_report is None:
                return False
            
            # Phase 5: Generate Outputs
            if not self._generate_outputs(processed_data, metrics_report):
                return False
            
            # Phase 6: Visualization (if visualization module is available)
            self._generate_visualizations(processed_data, metrics_report)
            
            self.logger.info("Complete analysis finished successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Analysis failed with error: {e}")
            self.experiment_logger.log_error(f"Analysis failed: {e}", e)
            return False
    
    def _validate_inputs(self) -> bool:
        """Validate input configuration and files.
        
        Returns:
            True if validation passed, False otherwise
        """
        self.experiment_logger.log_phase_start("input_validation")
        
        try:
            # Check for missing files
            missing_files = self.config.validate_required_files()
            if missing_files:
                error_msg = f"Missing required files: {missing_files}"
                self.logger.error(error_msg)
                self.experiment_logger.log_phase_completion(
                    "input_validation", 
                    {"status": "failed", "missing_files": missing_files}
                )
                return False
            
            self.experiment_logger.log_phase_completion(
                "input_validation",
                {"status": "success", "files_validated": len(self.config.required_input_files)}
            )
            return True
            
        except Exception as e:
            error_msg = f"Input validation failed: {e}"
            self.logger.error(error_msg)
            self.experiment_logger.log_error(error_msg, e)
            return False
    
    def _load_data(self) -> Optional[LoadedData]:
        """Load all required data files.
        
        Returns:
            LoadedData container or None if loading failed
        """
        self.experiment_logger.log_phase_start("data_loading")
        
        try:
            loaded_data = self.data_loader.load_all_data(
                self.config.input_directory,
                self.config.output_file_path
            )
            
            # Validate data consistency
            validation_results = self.data_loader.validate_data_consistency(loaded_data)
            
            if not validation_results["valid"]:
                error_msg = f"Data validation failed: {validation_results['issues']}"
                self.logger.error(error_msg)
                self.experiment_logger.log_phase_completion(
                    "data_loading",
                    {"status": "failed", "validation_issues": validation_results["issues"]}
                )
                return None
            
            self.experiment_logger.log_phase_completion(
                "data_loading",
                {
                    "status": "success",
                    "statistics": validation_results["statistics"]
                }
            )
            
            return loaded_data
            
        except Exception as e:
            error_msg = f"Data loading failed: {e}"
            self.logger.error(error_msg)
            self.experiment_logger.log_error(error_msg, e)
            self.experiment_logger.log_phase_completion(
                "data_loading",
                {"status": "failed", "error": str(e)}
            )
            return None
    
    def _process_data(self, loaded_data: LoadedData) -> Optional[ProcessedData]:
        """Process loaded data into analysis-ready format.
        
        Args:
            loaded_data: Raw loaded data
            
        Returns:
            ProcessedData container or None if processing failed
        """
        self.experiment_logger.log_phase_start("data_processing")
        
        try:
            processed_data = self.data_processor.process_all_data(loaded_data)
            
            # Log processing statistics
            processing_stats = {
                "total_test_pairs": len(processed_data.all_test_pairs),
                "test_timestamps": len(processed_data.test_timestamps),
                "total_possible_pairs": processed_data.total_possible_pairs,
                "stability_bins_available": processed_data.stability_bins is not None
            }
            
            self.experiment_logger.log_phase_completion(
                "data_processing",
                {"status": "success", "statistics": processing_stats}
            )
            
            return processed_data
            
        except Exception as e:
            error_msg = f"Data processing failed: {e}"
            self.logger.error(error_msg)
            self.experiment_logger.log_error(error_msg, e)
            self.experiment_logger.log_phase_completion(
                "data_processing",
                {"status": "failed", "error": str(e)}
            )
            return None
    
    def _calculate_metrics(self, processed_data: ProcessedData) -> Optional[MetricsReport]:
        """Calculate comprehensive metrics.
        
        Args:
            processed_data: Processed analysis data
            
        Returns:
            MetricsReport or None if calculation failed
        """
        self.experiment_logger.log_phase_start("metrics_calculation")
        
        try:
            metrics_report = self.metrics_calculator.calculate_comprehensive_metrics(processed_data)
            
            # Log key metrics
            metrics_summary = {
                "model_f1": metrics_report.model_metrics.F1,
                "model_precision": metrics_report.model_metrics.Precision,
                "model_recall": metrics_report.model_metrics.Recall,
                "baseline_available": metrics_report.baseline_metrics is not None,
                "stability_bins_analyzed": len(metrics_report.metrics_by_stability)
            }
            
            self.experiment_logger.log_phase_completion(
                "metrics_calculation",
                {"status": "success", "summary": metrics_summary}
            )
            
            return metrics_report
            
        except Exception as e:
            error_msg = f"Metrics calculation failed: {e}"
            self.logger.error(error_msg)
            self.experiment_logger.log_error(error_msg, e)
            self.experiment_logger.log_phase_completion(
                "metrics_calculation",
                {"status": "failed", "error": str(e)}
            )
            return None
    
    def _generate_outputs(
        self,
        processed_data: ProcessedData,
        metrics_report: MetricsReport
    ) -> bool:
        """Generate output files and reports.
        
        Args:
            processed_data: Processed analysis data
            metrics_report: Calculated metrics
            
        Returns:
            True if outputs generated successfully, False otherwise
        """
        self.experiment_logger.log_phase_start("output_generation")
        
        try:
            # Generate metrics report file
            metrics_file_path = os.path.join(self.config.output_directory, "PerformanceMetrics.txt")
            self.metrics_calculator.write_metrics_report(metrics_report, metrics_file_path)
            
            # Generate ground truth CSV
            ground_truth_csv = self._generate_ground_truth_csv(processed_data)
            gt_csv_path = os.path.join(self.config.output_directory, "ground_truth.csv")
            result = self.file_manager.write_json_file(
                ground_truth_csv.to_dict('records'),
                gt_csv_path.replace('.csv', '.json')  # Use JSON for complex data
            )
            
            # Generate prediction CSV
            prediction_csv = self._generate_prediction_csv(processed_data)
            pred_csv_path = os.path.join(self.config.output_directory, "prediction.csv")
            result = self.file_manager.write_json_file(
                prediction_csv.to_dict('records'),
                pred_csv_path.replace('.csv', '.json')  # Use JSON for complex data
            )
            
            # Generate heatmap similarity score (if both CSVs are similar in structure)
            similarity_score_path = os.path.join(self.config.output_directory, "heatmap_similarity_score.txt")
            self._calculate_heatmap_similarity(ground_truth_csv, prediction_csv, similarity_score_path)
            
            outputs_generated = [
                "PerformanceMetrics.txt",
                "ground_truth.json",
                "prediction.json",
                "heatmap_similarity_score.txt"
            ]
            
            self.experiment_logger.log_phase_completion(
                "output_generation",
                {"status": "success", "files_generated": outputs_generated}
            )
            
            return True
            
        except Exception as e:
            error_msg = f"Output generation failed: {e}"
            self.logger.error(error_msg)
            self.experiment_logger.log_error(error_msg, e)
            self.experiment_logger.log_phase_completion(
                "output_generation",
                {"status": "failed", "error": str(e)}
            )
            return False
    
    def _generate_ground_truth_csv(self, processed_data: ProcessedData) -> Any:
        """Generate ground truth CSV data structure.
        
        Args:
            processed_data: Processed analysis data
            
        Returns:
            Ground truth data structure
        """
        # Generate heatmap data similar to original result.py logic
        test_data = processed_data.test_set_processed
        
        if test_data.empty or 'pair_relation' not in test_data.columns:
            self.logger.warning("Insufficient data for ground truth CSV generation")
            return []
        
        # Count pair relations
        pair_relation_counts = test_data['pair_relation'].value_counts()
        
        heatmap_data = []
        for pair_relation, count in pair_relation_counts.items():
            parts = pair_relation.split('_')
            if len(parts) >= 3:
                residue_a = parts[0]
                residue_b = parts[1]
                relation = parts[2]
                
                heatmap_data.append({
                    'residue_a': residue_a,
                    'residue_b': residue_b,
                    'relation': relation,
                    'values': count
                })
        
        # Add frequency normalization
        if heatmap_data:
            values = [item['values'] for item in heatmap_data]
            min_val, max_val = min(values), max(values)
            
            for item in heatmap_data:
                if max_val > min_val:
                    item['freq'] = 1 + (item['values'] - min_val) / (max_val - min_val) * 99
                else:
                    item['freq'] = 1
                item['freq'] = round(item['freq'], 2)
        
        return heatmap_data
    
    def _generate_prediction_csv(self, processed_data: ProcessedData) -> Any:
        """Generate prediction CSV data structure.
        
        Args:
            processed_data: Processed analysis data
            
        Returns:
            Prediction data structure
        """
        # Generate prediction heatmap data
        output_data = processed_data.output_processed
        
        if output_data.empty or 'pair' not in output_data.columns:
            self.logger.warning("Insufficient data for prediction CSV generation")
            return []
        
        # Count pairs
        pair_counts = output_data['pair'].value_counts()
        
        heatmap_data = []
        for pair, count in pair_counts.items():
            parts = pair.split('_')
            if len(parts) >= 2:
                residue_a = parts[0]
                residue_b = '_'.join(parts[1:])  # Handle multi-part residue names
                
                heatmap_data.append({
                    'residue_a': residue_a,
                    'residue_b': residue_b,
                    'values': count
                })
        
        # Add frequency normalization
        if heatmap_data:
            values = [item['values'] for item in heatmap_data]
            min_val, max_val = min(values), max(values)
            
            for item in heatmap_data:
                if max_val > min_val:
                    item['freq'] = 1 + (item['values'] - min_val) / (max_val - min_val) * 99
                else:
                    item['freq'] = 1
                item['freq'] = round(item['freq'], 2)
        
        return heatmap_data
    
    def _calculate_heatmap_similarity(self, gt_data: Any, pred_data: Any, output_file: str) -> None:
        """Calculate heatmap similarity score.
        
        Args:
            gt_data: Ground truth data
            pred_data: Prediction data
            output_file: Output file path
        """
        try:
            # Simple similarity calculation
            # This is a simplified version - the original had complex pivot table logic
            
            gt_pairs = set()
            pred_pairs = set()
            
            if isinstance(gt_data, list):
                gt_pairs = {f"{item['residue_a']}_{item['residue_b']}" for item in gt_data}
            
            if isinstance(pred_data, list):
                pred_pairs = {f"{item['residue_a']}_{item['residue_b']}" for item in pred_data}
            
            intersection = gt_pairs & pred_pairs
            union = gt_pairs | pred_pairs
            
            if len(union) > 0:
                jaccard_similarity = len(intersection) / len(union)
            else:
                jaccard_similarity = 0.0
            
            with open(output_file, "w") as f:
                f.write(f"Ground truth pairs: {len(gt_pairs)}\n")
                f.write(f"Prediction pairs: {len(pred_pairs)}\n")
                f.write(f"Intersection: {len(intersection)}\n")
                f.write(f"Union: {len(union)}\n")
                f.write(f"Jaccard similarity: {jaccard_similarity:.4f}\n")
            
            self.logger.info(f"Heatmap similarity score written to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to calculate heatmap similarity: {e}")
    
    def _generate_visualizations(
        self,
        processed_data: ProcessedData,
        metrics_report: MetricsReport
    ) -> None:
        """Generate visualizations (placeholder for future implementation).
        
        Args:
            processed_data: Processed analysis data
            metrics_report: Calculated metrics
        """
        self.experiment_logger.log_phase_start("visualization_generation")
        
        try:
            # Placeholder for visualization generation
            # The original result.py had extensive plotting logic that would be
            # implemented in separate visualization classes
            
            self.logger.info("Visualization generation placeholder - implement visualization classes")
            
            # Future: Import and use visualization classes
            # from .visualization import HeatmapPlotter, MetricsPlotter, TrajectoryPlotter
            
            self.experiment_logger.log_phase_completion(
                "visualization_generation",
                {"status": "placeholder", "note": "Implement visualization classes"}
            )
            
        except Exception as e:
            self.logger.warning(f"Visualization generation failed: {e}")
            self.experiment_logger.log_phase_completion(
                "visualization_generation",
                {"status": "failed", "error": str(e)}
            )


def create_analysis_config_from_args() -> AnalysisConfig:
    """Create analysis configuration from command line arguments.
    
    Returns:
        AnalysisConfig instance
    """
    parser = argparse.ArgumentParser(description="RNN-MD Results Analysis")
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing train.txt, test.txt, and valid.txt files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to store the output files')
    parser.add_argument('--output_file_dir', type=str, required=True,
                       help='Path to the prediction output file')
    parser.add_argument('--num_pairs_to_show', type=int, default=50,
                       help='Number of pairs to show in visualizations')
    parser.add_argument('--valid_steps_to_show', type=int, default=20,
                       help='Number of validation steps to show')
    
    args = parser.parse_args()
    
    return AnalysisConfig(
        input_directory=args.input_dir,
        output_directory=args.output_dir,
        output_file_path=args.output_file_dir,
        num_pairs_to_show=args.num_pairs_to_show,
        valid_steps_to_show=args.valid_steps_to_show
    )


def main():
    """Main entry point for results analysis."""
    try:
        # Create configuration from command line arguments
        config = create_analysis_config_from_args()
        
        # Create and run results manager
        results_manager = ResultsManager(config)
        success = results_manager.run_complete_analysis()
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1


if __name__ == "__main__":
    import sys
    exit_code = main()
    sys.exit(exit_code) 
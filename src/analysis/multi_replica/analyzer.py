"""Multi-replica analyzer for aggregating metrics across multiple replicas."""

import sys
from pathlib import Path

# Add src to path for absolute imports
sys.path.insert(0, str(Path(__file__).parents[2]))

import json
import glob
from typing import Dict, List, Optional, Union, Any
from collections import defaultdict

from core.utils import get_logger
from .data_structures import (
    AggregatedMetrics, OverallStats, StabilityGroupStats, 
    MetricStats, ReplicaInfo
)


class MultiReplicaAnalyzer:
    """Analyzes metrics across multiple replica runs."""
    
    def __init__(self):
        """Initialize multi-replica analyzer."""
        self.logger = get_logger(__name__)
        self.replicas: List[ReplicaInfo] = []
        self.raw_metrics: List[Dict[str, Any]] = []
    
    def load_replica_metrics(self, replica_paths: List[str]) -> bool:
        """Load metrics from multiple replica directories.
        
        Args:
            replica_paths: List of paths to analysis directories containing metrics_structured.json
            
        Returns:
            True if at least one replica loaded successfully
        """
        self.replicas.clear()
        self.raw_metrics.clear()
        
        self.logger.info(f"Loading metrics from {len(replica_paths)} replica directories")
        
        successful_loads = 0
        
        for replica_path in replica_paths:
            replica_info = ReplicaInfo(replica_path=replica_path)
            
            try:
                # Look for metrics_structured.json file
                metrics_file = Path(replica_path) / "metrics_structured.json"
                
                if not metrics_file.exists():
                    replica_info.load_success = False
                    replica_info.error_message = f"metrics_structured.json not found in {replica_path}"
                    self.logger.warning(replica_info.error_message)
                    self.replicas.append(replica_info)
                    continue
                
                # Load JSON file
                with open(metrics_file, 'r') as f:
                    metrics_data = json.load(f)
                
                # Validate structure
                required_keys = ['metadata', 'overall_metrics', 'stability_metrics']
                missing_keys = [key for key in required_keys if key not in metrics_data]
                
                if missing_keys:
                    replica_info.load_success = False
                    replica_info.error_message = f"Missing required keys: {missing_keys}"
                    self.logger.warning(f"Invalid structure in {metrics_file}: {replica_info.error_message}")
                    self.replicas.append(replica_info)
                    continue
                
                # Store metadata
                replica_info.metadata = metrics_data.get('metadata', {})
                replica_info.load_success = True
                
                # Store raw metrics
                self.raw_metrics.append(metrics_data)
                successful_loads += 1
                
                self.logger.info(f"Successfully loaded metrics from {replica_path}")
                
            except Exception as e:
                replica_info.load_success = False
                replica_info.error_message = f"Error loading {replica_path}: {str(e)}"
                self.logger.error(replica_info.error_message)
            
            self.replicas.append(replica_info)
        
        self.logger.info(f"Successfully loaded {successful_loads}/{len(replica_paths)} replica metrics")
        
        return successful_loads > 0
    
    def discover_replica_directories(self, base_path: str, pattern: str = "*/analysis") -> List[str]:
        """Discover replica analysis directories automatically.
        
        Args:
            base_path: Base directory to search in
            pattern: Glob pattern to find analysis directories
            
        Returns:
            List of discovered analysis directory paths
        """
        search_pattern = str(Path(base_path) / pattern)
        discovered_paths = glob.glob(search_pattern)
        
        # Filter to only directories that contain metrics_structured.json
        valid_paths = []
        for path in discovered_paths:
            metrics_file = Path(path) / "metrics_structured.json"
            if metrics_file.exists():
                valid_paths.append(path)
        
        self.logger.info(f"Discovered {len(valid_paths)} replica directories with structured metrics")
        return valid_paths
    
    def aggregate_metrics(self) -> Optional[AggregatedMetrics]:
        """Aggregate metrics across all loaded replicas.
        
        Returns:
            AggregatedMetrics object or None if aggregation fails
        """
        if not self.raw_metrics:
            self.logger.error("No raw metrics available. Load replica metrics first.")
            return None
        
        try:
            self.logger.info(f"Aggregating metrics across {len(self.raw_metrics)} replicas")
            
            # Aggregate overall metrics
            overall_stats = self._aggregate_overall_metrics()
            
            # Aggregate stability group metrics
            training_freq_stats = self._aggregate_stability_metrics("training_frequency")
            test_freq_stats = self._aggregate_stability_metrics("test_frequency")
            
            # Create aggregated metrics object
            aggregated = AggregatedMetrics(
                overall_stats=overall_stats,
                training_frequency_stats=training_freq_stats,
                test_frequency_stats=test_freq_stats,
                replica_paths=[r.replica_path for r in self.replicas if r.load_success],
                n_replicas=len(self.raw_metrics),
                experiment_name=self._extract_experiment_name()
            )
            
            self.logger.info("Successfully aggregated metrics across replicas")
            return aggregated
            
        except Exception as e:
            self.logger.error(f"Failed to aggregate metrics: {e}")
            return None
    
    def _aggregate_overall_metrics(self) -> OverallStats:
        """Aggregate overall model and baseline metrics."""
        
        # Collect model metrics
        model_metrics = defaultdict(list)
        baseline_metrics = defaultdict(list)
        
        for metrics in self.raw_metrics:
            overall = metrics['overall_metrics']
            
            # Model metrics
            model_data = overall['model']
            for metric_name, value in model_data.items():
                if isinstance(value, (int, float)) and metric_name in ['recall', 'precision', 'f1', 'mcc']:
                    model_metrics[f"model_{metric_name}"].append(float(value))
            
            # Model mean pairwise F1
            if 'mean_pairwise_f1' in overall:
                mpf1 = overall['mean_pairwise_f1']
                if isinstance(mpf1, dict) and 'model' in mpf1 and mpf1['model'] is not None:
                    model_metrics['model_mean_pairwise_f1'].append(float(mpf1['model']))
            
            # Baseline metrics (if available)
            if overall.get('baseline') is not None:
                baseline_data = overall['baseline']
                for metric_name, value in baseline_data.items():
                    if isinstance(value, (int, float)) and metric_name in ['recall', 'precision', 'f1', 'mcc']:
                        baseline_metrics[f"baseline_{metric_name}"].append(float(value))
                
                # Baseline mean pairwise F1
                if 'mean_pairwise_f1' in overall:
                    mpf1 = overall['mean_pairwise_f1']
                    if isinstance(mpf1, dict) and 'baseline' in mpf1 and mpf1['baseline'] is not None:
                        baseline_metrics['baseline_mean_pairwise_f1'].append(float(mpf1['baseline']))
        
        # Create MetricStats objects
        overall_stats = OverallStats(
            model_recall=MetricStats(values=model_metrics['model_recall'], mean=0, std=0, min=0, max=0),
            model_precision=MetricStats(values=model_metrics['model_precision'], mean=0, std=0, min=0, max=0),
            model_f1=MetricStats(values=model_metrics['model_f1'], mean=0, std=0, min=0, max=0),
            model_mcc=MetricStats(values=model_metrics['model_mcc'], mean=0, std=0, min=0, max=0),
            model_mean_pairwise_f1=MetricStats(values=model_metrics['model_mean_pairwise_f1'], mean=0, std=0, min=0, max=0),
            n_replicas=len(self.raw_metrics)
        )
        
        # Add baseline stats if available
        if baseline_metrics['baseline_recall']:
            overall_stats.baseline_recall = MetricStats(values=baseline_metrics['baseline_recall'], mean=0, std=0, min=0, max=0)
            overall_stats.baseline_precision = MetricStats(values=baseline_metrics['baseline_precision'], mean=0, std=0, min=0, max=0)
            overall_stats.baseline_f1 = MetricStats(values=baseline_metrics['baseline_f1'], mean=0, std=0, min=0, max=0)
            overall_stats.baseline_mcc = MetricStats(values=baseline_metrics['baseline_mcc'], mean=0, std=0, min=0, max=0)
        
        if baseline_metrics['baseline_mean_pairwise_f1']:
            overall_stats.baseline_mean_pairwise_f1 = MetricStats(values=baseline_metrics['baseline_mean_pairwise_f1'], mean=0, std=0, min=0, max=0)
        
        return overall_stats
    
    def _aggregate_stability_metrics(self, frequency_type: str) -> Dict[str, StabilityGroupStats]:
        """Aggregate stability group metrics for a given frequency type."""
        
        stability_data = defaultdict(lambda: defaultdict(list))
        
        # Collect metrics by stability group
        for metrics in self.raw_metrics:
            stability_metrics = metrics['stability_metrics'].get(frequency_type, {})
            
            for group_name, group_data in stability_metrics.items():
                if not isinstance(group_data, dict):
                    continue
                
                # Performance metrics
                for metric_name in ['recall', 'precision', 'f1', 'mcc', 'tpr', 'fpr']:
                    if metric_name in group_data and isinstance(group_data[metric_name], (int, float)):
                        stability_data[group_name][metric_name].append(float(group_data[metric_name]))
                
                # Additional metrics
                for metric_name in ['mean_pairwise_f1', 'pair_count']:
                    if metric_name in group_data and isinstance(group_data[metric_name], (int, float)):
                        stability_data[group_name][metric_name].append(float(group_data[metric_name]))
                
                # Baseline metrics (optional)
                if 'baseline_mean_pairwise_f1' in group_data and group_data['baseline_mean_pairwise_f1'] is not None:
                    stability_data[group_name]['baseline_mean_pairwise_f1'].append(float(group_data['baseline_mean_pairwise_f1']))
        
        # Create StabilityGroupStats objects
        aggregated_stability = {}
        
        for group_name, group_metrics in stability_data.items():
            group_stats = StabilityGroupStats(
                recall=MetricStats(values=group_metrics['recall'], mean=0, std=0, min=0, max=0),
                precision=MetricStats(values=group_metrics['precision'], mean=0, std=0, min=0, max=0),
                f1=MetricStats(values=group_metrics['f1'], mean=0, std=0, min=0, max=0),
                mcc=MetricStats(values=group_metrics['mcc'], mean=0, std=0, min=0, max=0),
                tpr=MetricStats(values=group_metrics['tpr'], mean=0, std=0, min=0, max=0),
                fpr=MetricStats(values=group_metrics['fpr'], mean=0, std=0, min=0, max=0),
                mean_pairwise_f1=MetricStats(values=group_metrics['mean_pairwise_f1'], mean=0, std=0, min=0, max=0),
                pair_count=MetricStats(values=group_metrics['pair_count'], mean=0, std=0, min=0, max=0),
                group_name=group_name,
                n_replicas=len(group_metrics['recall'])
            )
            
            # Add baseline if available
            if group_metrics['baseline_mean_pairwise_f1']:
                group_stats.baseline_mean_pairwise_f1 = MetricStats(values=group_metrics['baseline_mean_pairwise_f1'], mean=0, std=0, min=0, max=0)
            
            aggregated_stability[group_name] = group_stats
        
        return aggregated_stability
    
    def _extract_experiment_name(self) -> str:
        """Extract experiment name from replica metadata."""
        for replica in self.replicas:
            if replica.load_success and 'experiment_name' in replica.metadata:
                return replica.metadata['experiment_name']
        
        # Fallback: extract from first replica path
        if self.replicas:
            return Path(self.replicas[0].replica_path).parent.name.split('_')[0]
        
        return "multi_replica_experiment"
    
    def get_replica_summary(self) -> Dict[str, Any]:
        """Get summary of replica loading status.
        
        Returns:
            Dictionary with summary information
        """
        total = len(self.replicas)
        successful = sum(1 for r in self.replicas if r.load_success)
        failed = total - successful
        
        return {
            'total_replicas': total,
            'successful_loads': successful,
            'failed_loads': failed,
            'success_rate': successful / total if total > 0 else 0,
            'replica_details': [
                {
                    'path': r.replica_path,
                    'success': r.load_success,
                    'error': r.error_message if not r.load_success else None
                }
                for r in self.replicas
            ]
        }
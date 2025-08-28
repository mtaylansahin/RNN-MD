"""Multi-replica plotter for creating visualizations with error bars."""

import sys
from pathlib import Path

# Add src to path for absolute imports
sys.path.insert(0, str(Path(__file__).parents[2]))

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import os

from core.utils import get_logger
from .data_structures import AggregatedMetrics, StabilityGroupStats, MetricStats


class MultiReplicaPlotter:
    """Creates visualizations for multi-replica analysis with error bars."""
    
    def __init__(self, output_directory: str):
        """Initialize multi-replica plotter.
        
        Args:
            output_directory: Directory to save plots
        """
        self.output_directory = output_directory
        self.logger = get_logger(__name__)
        
        # Create output directory
        Path(output_directory).mkdir(parents=True, exist_ok=True)
        
        # Plot styling
        self.figure_size = (14, 10)
        self.dpi = 300
        self.colors = {
            'recall': '#2E86AB',
            'precision': '#A23B72', 
            'f1': '#F18F01',
            'mcc': '#C73E1D',
            'mean_pairwise_f1': '#36213E',
            'baseline_mean_pairwise_f1': '#7A9E7E'
        }
    
    def generate_all_plots(self, aggregated_metrics: AggregatedMetrics) -> List[str]:
        """Generate all multi-replica plots.
        
        Args:
            aggregated_metrics: Aggregated metrics across replicas
            
        Returns:
            List of generated plot file paths
        """
        self.logger.info("Generating multi-replica plots")
        
        generated_plots = []
        
        try:
            # Overall metrics comparison plot
            overall_plot = self.plot_overall_metrics(aggregated_metrics)
            if overall_plot:
                generated_plots.append(overall_plot)
            
            # Stability group metrics (training frequency)
            training_plot = self.plot_stability_metrics(
                aggregated_metrics, 
                frequency_type="training",
                title="Performance by Interaction Stability (Training Frequency)",
                filename="multi_replica_stability_training.png"
            )
            if training_plot:
                generated_plots.append(training_plot)
            
            # Stability group metrics (test frequency)
            test_plot = self.plot_stability_metrics(
                aggregated_metrics,
                frequency_type="test", 
                title="Performance by Interaction Stability (Test Frequency)",
                filename="multi_replica_stability_test.png"
            )
            if test_plot:
                generated_plots.append(test_plot)
            
            # Comparison plot showing training vs test frequency
            comparison_plot = self.plot_training_vs_test_comparison(aggregated_metrics)
            if comparison_plot:
                generated_plots.append(comparison_plot)
            
            # Individual metric plots with detailed error bars
            individual_plots = self.plot_individual_metrics(aggregated_metrics)
            generated_plots.extend(individual_plots)
            
            self.logger.info(f"Generated {len(generated_plots)} multi-replica plots")
            
        except Exception as e:
            self.logger.error(f"Failed to generate multi-replica plots: {e}")
        
        return generated_plots
    
    def plot_overall_metrics(self, aggregated_metrics: AggregatedMetrics) -> Optional[str]:
        """Plot overall model performance across replicas.
        
        Args:
            aggregated_metrics: Aggregated metrics data
            
        Returns:
            Path to generated plot file or None
        """
        try:
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
            
            overall = aggregated_metrics.overall_stats
            
            # Metrics to plot
            metrics = ['model_recall', 'model_precision', 'model_f1', 'model_mcc', 'model_mean_pairwise_f1']
            metric_labels = ['Recall', 'Precision', 'F1', 'MCC', 'Mean Pairwise F1']
            
            # Get values and error bars
            means = []
            stds = []
            colors = []
            
            for metric in metrics:
                metric_stats: MetricStats = getattr(overall, metric)
                means.append(metric_stats.mean)
                stds.append(metric_stats.std)
                colors.append(self.colors.get(metric.replace('model_', ''), '#333333'))
            
            # Create bar plot with error bars
            x_pos = np.arange(len(metric_labels))
            bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color=colors, alpha=0.7, 
                         edgecolor='white', linewidth=1)
            
            # Add value labels on bars
            for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                       f'{mean:.3f}±{std:.3f}',
                       ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            # Styling
            ax.set_xlabel('Metrics', fontsize=14, fontweight='bold')
            ax.set_ylabel('Score', fontsize=14, fontweight='bold')
            ax.set_title(f'Overall Model Performance Across {overall.n_replicas} Replicas', 
                        fontsize=16, fontweight='bold', pad=20)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(metric_labels, fontsize=12)
            ax.set_ylim(0, 1.0)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add replica count annotation
            ax.text(0.02, 0.98, f'N = {overall.n_replicas} replicas', 
                   transform=ax.transAxes, fontsize=12, 
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
                   verticalalignment='top')
            
            plt.tight_layout()
            
            # Save plot
            output_path = os.path.join(self.output_directory, "multi_replica_overall_metrics.png")
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Generated overall metrics plot: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate overall metrics plot: {e}")
            return None
    
    def plot_stability_metrics(
        self, 
        aggregated_metrics: AggregatedMetrics,
        frequency_type: str = "training",
        title: Optional[str] = None,
        filename: Optional[str] = None
    ) -> Optional[str]:
        """Plot stability group metrics with error bars.
        
        Args:
            aggregated_metrics: Aggregated metrics data
            frequency_type: Either "training" or "test"
            title: Custom plot title
            filename: Custom filename
            
        Returns:
            Path to generated plot file or None
        """
        try:
            stability_stats = aggregated_metrics.get_stability_groups(frequency_type)
            
            if not stability_stats:
                self.logger.warning(f"No stability metrics found for {frequency_type} frequency")
                return None
            
            # Get groups in standard order
            group_names = aggregated_metrics.get_group_names(frequency_type)
            
            # Filter to only existing groups  
            existing_groups = [name for name in group_names if name in stability_stats]
            
            if not existing_groups:
                self.logger.warning(f"No existing stability groups found for {frequency_type}")
                return None
            
            # Metrics to plot
            metric_names = ['recall', 'precision', 'f1', 'mcc', 'mean_pairwise_f1']
            metric_labels = ['Recall', 'Precision', 'F1', 'MCC', 'Mean Pairwise F1']
            
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
            
            # Set up bar positions
            n_groups = len(existing_groups)
            n_metrics = len(metric_names)
            bar_width = 0.15
            x_pos = np.arange(n_groups)
            
            # Plot each metric
            for i, (metric, label) in enumerate(zip(metric_names, metric_labels)):
                means = []
                stds = []
                pair_counts = []
                
                for group_name in existing_groups:
                    group_stats = stability_stats[group_name]
                    metric_stats: MetricStats = getattr(group_stats, metric)
                    means.append(metric_stats.mean)
                    stds.append(metric_stats.std)
                    pair_counts.append(int(group_stats.pair_count.mean))
                
                # Create bars with error bars
                x_offset = x_pos + (i - n_metrics/2) * bar_width
                bars = ax.bar(x_offset, means, bar_width, yerr=stds, capsize=3,
                             label=label, color=self.colors.get(metric, f'C{i}'), 
                             alpha=0.8, edgecolor='white', linewidth=0.5)
                
                # Add value labels (only for F1 to avoid clutter)
                if metric == 'f1':
                    for j, (bar, mean, std) in enumerate(zip(bars, means, stds)):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                               f'{mean:.3f}', ha='center', va='bottom', 
                               fontsize=8, fontweight='bold')
            
            # Styling
            ax.set_xlabel('Stability Groups', fontsize=14, fontweight='bold')
            ax.set_ylabel('Score', fontsize=14, fontweight='bold')
            
            if title is None:
                title = f'Performance by Interaction Stability ({frequency_type.title()} Frequency)'
            ax.set_title(f'{title}\nAcross {aggregated_metrics.n_replicas} Replicas', 
                        fontsize=16, fontweight='bold', pad=20)
            
            # X-axis labels with pair counts
            group_labels_with_counts = []
            for group_name in existing_groups:
                group_stats = stability_stats[group_name] 
                pair_count = int(group_stats.pair_count.mean)
                pair_count_std = int(group_stats.pair_count.std)
                if pair_count_std > 0:
                    count_str = f'(N={pair_count}±{pair_count_std})'
                else:
                    count_str = f'(N={pair_count})'
                
                # Clean up group name
                clean_name = group_name.replace('Moderate', 'Uncommon')
                if group_name == 'Undefined':
                    clean_name = 'Not in Train' if frequency_type == 'training' else 'Not in Test'
                
                group_labels_with_counts.append(f'{clean_name}\n{count_str}')
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(group_labels_with_counts, fontsize=11)
            ax.set_ylim(0, 1.0)
            
            # Legend
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
            
            # Grid
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add replica count annotation
            ax.text(0.02, 0.98, f'N = {aggregated_metrics.n_replicas} replicas', 
                   transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
                   verticalalignment='top')
            
            plt.tight_layout()
            
            # Save plot
            if filename is None:
                filename = f"multi_replica_stability_{frequency_type}.png"
            output_path = os.path.join(self.output_directory, filename)
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Generated stability metrics plot: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate stability metrics plot: {e}")
            return None
    
    def plot_training_vs_test_comparison(self, aggregated_metrics: AggregatedMetrics) -> Optional[str]:
        """Plot comparison between training and test frequency stability metrics.
        
        Args:
            aggregated_metrics: Aggregated metrics data
            
        Returns:
            Path to generated plot file or None
        """
        try:
            training_stats = aggregated_metrics.training_frequency_stats
            test_stats = aggregated_metrics.test_frequency_stats
            
            if not training_stats or not test_stats:
                self.logger.warning("Missing training or test stability stats for comparison")
                return None
            
            # Find common groups
            common_groups = set(training_stats.keys()).intersection(set(test_stats.keys()))
            common_groups = [g for g in ['Rare (<5%)', 'Moderate (5-50%)', 'Stable (>50%)', 'Undefined'] 
                           if g in common_groups]
            
            if not common_groups:
                self.logger.warning("No common stability groups found for comparison")
                return None
            
            # Create subplot for F1 scores comparison
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), dpi=self.dpi)
            
            # Plot 1: F1 Score comparison
            x_pos = np.arange(len(common_groups))
            bar_width = 0.35
            
            training_f1_means = [training_stats[g].f1.mean for g in common_groups]
            training_f1_stds = [training_stats[g].f1.std for g in common_groups]
            test_f1_means = [test_stats[g].f1.mean for g in common_groups]
            test_f1_stds = [test_stats[g].f1.std for g in common_groups]
            
            bars1 = ax1.bar(x_pos - bar_width/2, training_f1_means, bar_width, 
                           yerr=training_f1_stds, capsize=5, label='Training Frequency',
                           color=self.colors['f1'], alpha=0.7, edgecolor='white')
            
            bars2 = ax1.bar(x_pos + bar_width/2, test_f1_means, bar_width,
                           yerr=test_f1_stds, capsize=5, label='Test Frequency', 
                           color='#FF6B6B', alpha=0.7, edgecolor='white')
            
            # Add value labels
            for bars, means in [(bars1, training_f1_means), (bars2, test_f1_means)]:
                for bar, mean in zip(bars, means):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{mean:.3f}', ha='center', va='bottom', 
                           fontsize=9, fontweight='bold')
            
            ax1.set_xlabel('Stability Groups', fontsize=12, fontweight='bold')
            ax1.set_ylabel('F1 Score', fontsize=12, fontweight='bold') 
            ax1.set_title('F1 Score: Training vs Test Frequency Binning', fontsize=14, fontweight='bold')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels([g.replace('Moderate', 'Uncommon') for g in common_groups])
            ax1.legend()
            ax1.grid(True, alpha=0.3, axis='y')
            ax1.set_ylim(0, 1.0)
            
            # Plot 2: Pair Count comparison  
            training_counts = [training_stats[g].pair_count.mean for g in common_groups]
            training_count_stds = [training_stats[g].pair_count.std for g in common_groups]
            test_counts = [test_stats[g].pair_count.mean for g in common_groups]
            test_count_stds = [test_stats[g].pair_count.std for g in common_groups]
            
            bars3 = ax2.bar(x_pos - bar_width/2, training_counts, bar_width,
                           yerr=training_count_stds, capsize=5, label='Training Frequency',
                           color='#4ECDC4', alpha=0.7, edgecolor='white')
            
            bars4 = ax2.bar(x_pos + bar_width/2, test_counts, bar_width,
                           yerr=test_count_stds, capsize=5, label='Test Frequency',
                           color='#45B7D1', alpha=0.7, edgecolor='white')
            
            ax2.set_xlabel('Stability Groups', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Pair Count', fontsize=12, fontweight='bold')
            ax2.set_title('Pair Counts: Training vs Test Frequency Binning', fontsize=14, fontweight='bold') 
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels([g.replace('Moderate', 'Uncommon') for g in common_groups])
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')
            
            plt.suptitle(f'Training vs Test Frequency Comparison\nAcross {aggregated_metrics.n_replicas} Replicas', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save plot
            output_path = os.path.join(self.output_directory, "multi_replica_training_vs_test_comparison.png")
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Generated training vs test comparison plot: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate training vs test comparison plot: {e}")
            return None
    
    def plot_individual_metrics(self, aggregated_metrics: AggregatedMetrics) -> List[str]:
        """Plot individual metrics in detail with error bars.
        
        Args:
            aggregated_metrics: Aggregated metrics data
            
        Returns:
            List of generated plot file paths
        """
        generated_plots = []
        
        metrics_to_plot = [
            ('f1', 'F1 Score'),
            ('recall', 'Recall'),
            ('precision', 'Precision'),
            ('mcc', 'MCC')
        ]
        
        for metric_name, metric_label in metrics_to_plot:
            plot_path = self._plot_single_metric_comparison(
                aggregated_metrics, metric_name, metric_label
            )
            if plot_path:
                generated_plots.append(plot_path)
        
        return generated_plots
    
    def _plot_single_metric_comparison(
        self, 
        aggregated_metrics: AggregatedMetrics, 
        metric_name: str, 
        metric_label: str
    ) -> Optional[str]:
        """Plot a single metric comparing training vs test frequency binning."""
        try:
            fig, ax = plt.subplots(figsize=(12, 8), dpi=self.dpi)
            
            training_stats = aggregated_metrics.training_frequency_stats
            test_stats = aggregated_metrics.test_frequency_stats
            
            # Find common groups
            common_groups = set(training_stats.keys()).intersection(set(test_stats.keys()))
            common_groups = [g for g in ['Rare (<5%)', 'Moderate (5-50%)', 'Stable (>50%)', 'Undefined']
                           if g in common_groups]
            
            if not common_groups:
                return None
            
            x_pos = np.arange(len(common_groups))
            bar_width = 0.35
            
            # Get metric values
            training_means = []
            training_stds = []
            test_means = []
            test_stds = []
            
            for group in common_groups:
                training_metric_stats = getattr(training_stats[group], metric_name)
                test_metric_stats = getattr(test_stats[group], metric_name)
                
                training_means.append(training_metric_stats.mean)
                training_stds.append(training_metric_stats.std)
                test_means.append(test_metric_stats.mean)
                test_stds.append(test_metric_stats.std)
            
            # Create bars
            bars1 = ax.bar(x_pos - bar_width/2, training_means, bar_width,
                          yerr=training_stds, capsize=5, label='Training Frequency',
                          color=self.colors.get(metric_name, '#4ECDC4'), alpha=0.7)
            
            bars2 = ax.bar(x_pos + bar_width/2, test_means, bar_width,
                          yerr=test_stds, capsize=5, label='Test Frequency',
                          color='#FF6B6B', alpha=0.7)
            
            # Styling
            ax.set_xlabel('Stability Groups', fontsize=14, fontweight='bold')
            ax.set_ylabel(metric_label, fontsize=14, fontweight='bold')
            ax.set_title(f'{metric_label} by Stability Group\nAcross {aggregated_metrics.n_replicas} Replicas',
                        fontsize=16, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels([g.replace('Moderate', 'Uncommon') for g in common_groups])
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, 1.0)
            
            plt.tight_layout()
            
            # Save plot
            output_path = os.path.join(self.output_directory, f"multi_replica_{metric_name}_comparison.png")
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Generated {metric_label} comparison plot: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate {metric_label} comparison plot: {e}")
            return None
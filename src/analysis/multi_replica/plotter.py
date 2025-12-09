"""Multi-replica plotter for creating visualizations with error bars."""

import sys
from pathlib import Path

# Add src to path for absolute imports
sys.path.insert(0, str(Path(__file__).parents[2]))

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import os

# Set numpy seed for reproducible jitter
np.random.seed(42)

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
        
        # Plot styling - Publication Ready (Okabe-Ito & Guide)
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'],
            'font.size': 20,
            'axes.titlesize': 26,
            'axes.labelsize': 22,
            'xtick.labelsize': 20,
            'ytick.labelsize': 20,
            'legend.fontsize': 22,
            'figure.titlesize': 32,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'axes.linewidth': 1.5,
            'axes.edgecolor': '#2C3E50',
            'axes.grid': True,
            'grid.alpha': 0.2,
            'grid.linewidth': 0.6,
            'grid.color': '#BDC3C7',
            'axes.axisbelow': True,
            'legend.frameon': False,
            'legend.fancybox': True,
            'legend.borderpad': 0.5,
        })
        
        self.figure_size = (14, 10)
        self.dpi = 300
        
        # Okabe-Ito Colorblind-Safe Palette
        self.okabe_ito = {
            'vermilion': '#D55E00',
            'blue': '#0072B2',
            'bluish_green': '#009E73',
            'orange': '#E69F00',
            'sky_blue': '#56B4E9',
            'reddish_purple': '#CC79A7',
            'yellow': '#F0E442',
            'black': '#000000'
        }

        # Stability Group Colors (Okabe-Ito)
        self.stability_palette = {
            'Rare': self.okabe_ito['vermilion'],      # #D55E00
            'Transient': self.okabe_ito['orange'],    # #E69F00
            'Stable': self.okabe_ito['bluish_green']  # #009E73
        }
        
        # Publication utility colors
        self.publication_colors = {
            'primary': '#2C3E50',
            'secondary': '#34495E',
            'background': '#ECF0F1',
            'grid': '#BDC3C7',
            'highlight': '#3498DB'
        }
        
        # Replica colors (Okabe-Ito sequence)
        self.replica_colors = [
            self.okabe_ito['blue'], 
            self.okabe_ito['vermilion'], 
            self.okabe_ito['bluish_green'], 
            self.okabe_ito['orange'], 
            self.okabe_ito['sky_blue'], 
            self.okabe_ito['reddish_purple'], 
            self.okabe_ito['yellow'], 
            self.okabe_ito['black']
        ]
        
        # Metric Colors mapped to Okabe-Ito
        self.colors = {
            'recall': self.okabe_ito['blue'],            # #0072B2
            'precision': self.okabe_ito['reddish_purple'],# #CC79A7
            'f1': self.okabe_ito['vermilion'],           # #D55E00
            'mcc': self.okabe_ito['sky_blue'],           # #56B4E9
            'mean_pairwise_f1': self.okabe_ito['bluish_green'], # #009E73
            'baseline_f1': '#7A9E7E',
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
            
            # Metrics to plot (include baseline if available)
            metrics = ['model_recall', 'model_precision', 'model_f1', 'model_mcc', 'model_mean_pairwise_f1']
            metric_labels = ['Recall', 'Precision', 'F1', 'MCC', 'Mean Pairwise F1']

            # Determine if baseline metrics are available
            baseline_available = any([
                overall.baseline_recall is not None,
                overall.baseline_precision is not None,
                overall.baseline_f1 is not None,
                overall.baseline_mcc is not None,
                overall.baseline_mean_pairwise_f1 is not None
            ])
            
            # Get values and error bars for model (and baseline if available)
            model_means = []
            model_stds = []
            model_colors = []

            for metric in metrics:
                metric_stats: MetricStats = getattr(overall, metric)
                model_means.append(metric_stats.mean)
                model_stds.append(metric_stats.std)
                model_colors.append(self.colors.get(metric.replace('model_', ''), '#333333'))

            # Baseline series (optional)
            baseline_means = []
            baseline_stds = []
            baseline_colors = []
            if baseline_available:
                for metric in metrics:
                    base_attr = metric.replace('model_', 'baseline_')
                    base_stats: Optional[MetricStats] = getattr(overall, base_attr, None)
                    if base_stats is not None:
                        baseline_means.append(base_stats.mean)
                        baseline_stds.append(base_stats.std)
                    else:
                        baseline_means.append(0.0)
                        baseline_stds.append(0.0)
                    baseline_colors.append(self.colors.get(base_attr.replace('baseline_', 'baseline_'), '#7A9E7E'))

            # Create grouped bar plot with error bars
            x_pos = np.arange(len(metric_labels))
            bar_width = 0.35 if baseline_available else 0.7
            bars_model = ax.bar(
                x_pos - (bar_width/2 if baseline_available else 0),
                model_means,
                bar_width,
                yerr=model_stds,
                capsize=5,
                color=model_colors,
                alpha=0.85,
                edgecolor='white',
                linewidth=1,
                label='Model'
            )

            bars_baseline = None
            if baseline_available:
                bars_baseline = ax.bar(
                    x_pos + bar_width/2,
                    baseline_means,
                    bar_width,
                    yerr=baseline_stds,
                    capsize=5,
                    color='#7A9E7E',
                    alpha=0.55,
                    edgecolor='white',
                    linewidth=1,
                    label='Baseline'
                )
                # Apply hatch pattern to baseline bars to avoid relying on colors
                for rect in bars_baseline:
                    rect.set_hatch('//')
            
            # Add individual datapoints using values from MetricStats
            for i, metric in enumerate(metrics):
                metric_stats: MetricStats = getattr(overall, metric)
                if metric_stats.values and len(metric_stats.values) > 1:
                    jitter = np.random.normal(0, 0.05, len(metric_stats.values))
                    x_center = x_pos[i] - (bar_width/2 if baseline_available else 0)
                    x_jittered = np.full(len(metric_stats.values), x_center) + jitter
                    ax.scatter(x_jittered, metric_stats.values, color='white', 
                               s=40, alpha=0.9, edgecolors=model_colors[i], linewidth=1.5, zorder=3)
                if baseline_available:
                    base_attr = metric.replace('model_', 'baseline_')
                    base_stats: Optional[MetricStats] = getattr(overall, base_attr, None)
                    if base_stats is not None and base_stats.values and len(base_stats.values) > 1:
                        jitter = np.random.normal(0, 0.05, len(base_stats.values))
                        x_center = x_pos[i] + bar_width/2
                        x_jittered = np.full(len(base_stats.values), x_center) + jitter
                        ax.scatter(
                            x_jittered,
                            base_stats.values,
                            color='white',
                            s=40,
                            alpha=0.9,
                            edgecolors='#7A9E7E',
                            linewidth=1.5,
                            zorder=3,
                            marker='s'  # square markers for baseline for pattern-based distinction
                        )
            
            # Add value labels on bars
            for i, (bar, mean, std) in enumerate(zip(bars_model, model_means, model_stds)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                        f'{mean:.3f}', ha='center', va='bottom', 
                        fontweight='bold', fontsize=12)
            if baseline_available and bars_baseline is not None:
                for i, (bar, mean, std) in enumerate(zip(bars_baseline, baseline_means, baseline_stds)):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                            f'{mean:.3f}', ha='center', va='bottom', 
                            fontweight='bold', fontsize=12, color='#2C3E50')
            
            # Styling
            ax.set_ylabel('Score', fontweight='bold')
            # Remove redundant X label if metrics are clear
            # ax.set_xlabel('Metrics', fontweight='bold') 
            
            title_suffix = f'Across {overall.n_replicas} Replicas'
            ax.set_title(f'Overall Performance {title_suffix}', fontweight='bold', pad=20)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(metric_labels)
            ax.set_ylim(0, 1.0)
            
            # Grid - horizontal only
            ax.grid(visible=True, axis='y', alpha=0.2)
            ax.grid(visible=False, axis='x')
            
            # Add replica count annotation
            stats_text = f"N = {overall.n_replicas} replicas"
            props = dict(boxstyle='round,pad=0.25', facecolor='white', 
                        alpha=0.85, edgecolor='gray', linewidth=0.5)
            ax.text(0.98, 0.98, stats_text, 
                   transform=ax.transAxes, fontsize=18,
                   bbox=props,
                   verticalalignment='top', horizontalalignment='right')
            
            if baseline_available:
                # Unified Legend
                handles = [
                    Patch(facecolor='gray', edgecolor='none', alpha=0.5, label='Model'),
                    Patch(facecolor='gray', edgecolor='none', hatch='//', alpha=0.5, label='Baseline')
                ]
                fig.legend(handles=handles, loc='lower center', 
                          bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False)
                plt.subplots_adjust(bottom=0.15)
            
            plt.tight_layout()
            if baseline_available:
                 plt.subplots_adjust(bottom=0.15) # Adjust again after tight_layout
            
            # Save plot
            output_path = os.path.join(self.output_directory, "multi_replica_overall_metrics.png")
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            
            # Also save SVG
            svg_path = os.path.join(self.output_directory, "multi_replica_overall_metrics.svg")
            plt.savefig(svg_path, format='svg', bbox_inches='tight')
            
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
            
            # Always include the three main stability categories
            required_categories = ['Rare (<5%)', 'Moderate (5-50%)', 'Stable (>50%)']
            
            # Use only the required categories to ensure consistency
            existing_groups = required_categories
            
            # Metrics to plot
            metric_names = ['recall', 'precision', 'f1', 'mcc', 'mean_pairwise_f1']
            metric_labels = ['Recall', 'Precision', 'F1', 'MCC', 'Mean Pairwise F1']
            
            # Check if any stability group has baseline F1 data, and if so, add it to metrics
            has_baseline_f1 = any(
                hasattr(group_stats, 'baseline_f1') and group_stats.baseline_f1 is not None
                for group_stats in stability_stats.values()
            )
            
            if has_baseline_f1:
                metric_names.append('baseline_f1')
                metric_labels.append('Baseline F1')
            
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
                    if group_name in stability_stats:
                        group_stats = stability_stats[group_name]
                        if metric == 'baseline_f1':
                            # Handle baseline_f1 specially since it might be None
                            baseline_f1_stats = getattr(group_stats, 'baseline_f1', None)
                            if baseline_f1_stats is not None:
                                means.append(baseline_f1_stats.mean)
                                stds.append(baseline_f1_stats.std)
                            else:
                                means.append(0.0)
                                stds.append(0.0)
                        else:
                            metric_stats: MetricStats = getattr(group_stats, metric)
                            means.append(metric_stats.mean)
                            stds.append(metric_stats.std)
                        pair_counts.append(int(group_stats.pair_count.mean))
                    else:
                        # Group doesn't exist, use zero values
                        means.append(0.0)
                        stds.append(0.0)
                        pair_counts.append(0)
                
                # Create bars with error bars
                x_offset = x_pos + (i - n_metrics/2) * bar_width
                bars = ax.bar(x_offset, means, bar_width, yerr=stds, capsize=3,
                             label=label, color=self.colors.get(metric, f'C{i}'), 
                             alpha=0.8, edgecolor='white', linewidth=0.5)
                
                # Add individual datapoints using values from MetricStats
                for j, group_name in enumerate(existing_groups):
                    if group_name in stability_stats:
                        group_stats = stability_stats[group_name]
                        if metric == 'baseline_f1':
                            # Handle baseline_f1 specially since it might be None
                            baseline_f1_stats = getattr(group_stats, 'baseline_f1', None)
                            if baseline_f1_stats is not None and baseline_f1_stats.values and len(baseline_f1_stats.values) > 1:
                                jitter = np.random.normal(0, 0.02, len(baseline_f1_stats.values))
                                x_jittered = np.full(len(baseline_f1_stats.values), x_offset[j]) + jitter
                                ax.scatter(x_jittered, baseline_f1_stats.values, 
                                         color=self.colors.get(metric, f'C{i}'), 
                                         s=30, alpha=0.9, edgecolors='white', linewidth=1, zorder=3)
                        else:
                            metric_stats: MetricStats = getattr(group_stats, metric)
                            if metric_stats.values and len(metric_stats.values) > 1:  # Only show points if we have multiple replicas
                                # Add small random jitter to x-position for visibility
                                jitter = np.random.normal(0, 0.02, len(metric_stats.values))
                                x_jittered = np.full(len(metric_stats.values), x_offset[j]) + jitter
                                
                                # Plot individual points
                                ax.scatter(x_jittered, metric_stats.values, 
                                         color=self.colors.get(metric, f'C{i}'), 
                                         s=30, alpha=0.9, edgecolors='white', linewidth=1, zorder=3)
                
                # Add value labels to all bars
                for j, (bar, mean, std) in enumerate(zip(bars, means, stds)):
                    if mean > 0:  # Only show labels for non-zero values
                        height = bar.get_height()
                        # Adjust font size based on bar height and metric type
                        fontsize = 7 if metric in ['recall', 'precision', 'mcc', 'mean_pairwise_f1'] else 8
                        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                               f'{mean:.3f}', ha='center', va='bottom', 
                               fontsize=fontsize, fontweight='bold', alpha=0.9)
            
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
                if group_name in stability_stats:
                    group_stats = stability_stats[group_name] 
                    pair_count_mean = group_stats.pair_count.mean
                    pair_count_std = group_stats.pair_count.std
                    
                    # Round mean to integer, but be more careful with std
                    pair_count = int(round(pair_count_mean))
                    
                    # Show std if it's meaningful (> 0.5) or if there are multiple replicas
                    if pair_count_std >= 0.5 or aggregated_metrics.n_replicas > 1:
                        # Round std but ensure it shows at least 1 if there's any variation
                        std_rounded = max(1, int(round(pair_count_std))) if pair_count_std > 0 else 0
                        if std_rounded > 0:
                            count_str = f'(N={pair_count}±{std_rounded})'
                        else:
                            count_str = f'(N={pair_count})'
                    else:
                        count_str = f'(N={pair_count})'
                else:
                    # Group doesn't exist, show zero count
                    count_str = '(N=0)'
                
                # Clean up group name
                clean_name = group_name.replace('Moderate', 'Transient')
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
            
            # Always use the three main required categories for consistency
            required_categories = ['Rare (<5%)', 'Moderate (5-50%)', 'Stable (>50%)']
            common_groups = required_categories
            
            # Create subplot for F1 scores comparison
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), dpi=self.dpi)
            
            # Plot 1: F1 Score comparison
            x_pos = np.arange(len(common_groups))
            bar_width = 0.35
            
            training_f1_means = [training_stats[g].f1.mean if g in training_stats else 0.0 for g in common_groups]
            training_f1_stds = [training_stats[g].f1.std if g in training_stats else 0.0 for g in common_groups]
            test_f1_means = [test_stats[g].f1.mean if g in test_stats else 0.0 for g in common_groups]
            test_f1_stds = [test_stats[g].f1.std if g in test_stats else 0.0 for g in common_groups]
            
            bars1 = ax1.bar(x_pos - bar_width/2, training_f1_means, bar_width, 
                           yerr=training_f1_stds, capsize=5, label='Training Frequency',
                           color=self.colors['f1'], alpha=0.7, edgecolor='white')
            
            bars2 = ax1.bar(x_pos + bar_width/2, test_f1_means, bar_width,
                           yerr=test_f1_stds, capsize=5, label='Test Frequency', 
                           color=self.stability_palette['Rare'], alpha=0.7, edgecolor='white')
            
            # Add individual datapoints for F1 scores using MetricStats.values
            for j, group_name in enumerate(common_groups):
                # Plot training frequency points
                if group_name in training_stats:
                    training_f1_stats = training_stats[group_name].f1
                    if training_f1_stats.values and len(training_f1_stats.values) > 1:
                        jitter = np.random.normal(0, 0.02, len(training_f1_stats.values))
                        x_jittered = np.full(len(training_f1_stats.values), x_pos[j] - bar_width/2) + jitter
                        ax1.scatter(x_jittered, training_f1_stats.values, 
                                  color=self.colors['f1'], s=30, alpha=0.9, 
                                  edgecolors='white', linewidth=1, zorder=3)
                
                # Plot test frequency points
                if group_name in test_stats:
                    test_f1_stats = test_stats[group_name].f1
                    if test_f1_stats.values and len(test_f1_stats.values) > 1:
                        jitter = np.random.normal(0, 0.02, len(test_f1_stats.values))
                        x_jittered = np.full(len(test_f1_stats.values), x_pos[j] + bar_width/2) + jitter
                        ax1.scatter(x_jittered, test_f1_stats.values, 
                                  color=self.stability_palette['Rare'], s=30, alpha=0.9, 
                                  edgecolors='white', linewidth=1, zorder=3)
            
            # Add value labels
            for bars, means in [(bars1, training_f1_means), (bars2, test_f1_means)]:
                for bar, mean in zip(bars, means):
                    if mean > 0:  # Only show labels for non-zero values
                        height = bar.get_height()
                        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{mean:.3f}', ha='center', va='bottom', 
                               fontsize=9, fontweight='bold')
            
            ax1.set_xlabel('Stability Groups', fontsize=12, fontweight='bold')
            ax1.set_ylabel('F1 Score', fontsize=12, fontweight='bold') 
            ax1.set_title('F1 Score: Training vs Test Frequency Binning', fontsize=14, fontweight='bold')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels([g.replace('Moderate', 'Transient') for g in common_groups])
            ax1.legend()
            ax1.grid(True, alpha=0.3, axis='y')
            ax1.set_ylim(0, 1.0)
            
            # Plot 2: Pair Count comparison  
            training_counts = [training_stats[g].pair_count.mean if g in training_stats else 0.0 for g in common_groups]
            training_count_stds = [training_stats[g].pair_count.std if g in training_stats else 0.0 for g in common_groups]
            test_counts = [test_stats[g].pair_count.mean if g in test_stats else 0.0 for g in common_groups]
            test_count_stds = [test_stats[g].pair_count.std if g in test_stats else 0.0 for g in common_groups]
            
            bars3 = ax2.bar(x_pos - bar_width/2, training_counts, bar_width,
                           yerr=training_count_stds, capsize=5, label='Training Frequency',
                           color=self.stability_palette['Stable'], alpha=0.7, edgecolor='white')
            
            bars4 = ax2.bar(x_pos + bar_width/2, test_counts, bar_width,
                           yerr=test_count_stds, capsize=5, label='Test Frequency',
                           color=self.publication_colors['highlight'], alpha=0.7, edgecolor='white')
            
            # Add individual datapoints for pair counts using MetricStats.values
            for j, group_name in enumerate(common_groups):
                # Plot training pair count points
                if group_name in training_stats:
                    training_count_stats = training_stats[group_name].pair_count
                    if training_count_stats.values and len(training_count_stats.values) > 1:
                        jitter = np.random.normal(0, 0.02, len(training_count_stats.values))
                        x_jittered = np.full(len(training_count_stats.values), x_pos[j] - bar_width/2) + jitter
                        ax2.scatter(x_jittered, training_count_stats.values, 
                                  color=self.stability_palette['Stable'], s=30, alpha=0.9, 
                                  edgecolors='white', linewidth=1, zorder=3)
                
                # Plot test pair count points
                if group_name in test_stats:
                    test_count_stats = test_stats[group_name].pair_count
                    if test_count_stats.values and len(test_count_stats.values) > 1:
                        jitter = np.random.normal(0, 0.02, len(test_count_stats.values))
                        x_jittered = np.full(len(test_count_stats.values), x_pos[j] + bar_width/2) + jitter
                        ax2.scatter(x_jittered, test_count_stats.values, 
                                  color=self.publication_colors['highlight'], s=30, alpha=0.9, 
                                  edgecolors='white', linewidth=1, zorder=3)
            
            ax2.set_xlabel('Stability Groups', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Pair Count', fontsize=12, fontweight='bold')
            ax2.set_title('Pair Counts: Training vs Test Frequency Binning', fontsize=14, fontweight='bold') 
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels([g.replace('Moderate', 'Transient') for g in common_groups])
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
            
            # Always use the three main required categories for consistency
            required_categories = ['Rare (<5%)', 'Moderate (5-50%)', 'Stable (>50%)']
            common_groups = required_categories
            
            x_pos = np.arange(len(common_groups))
            bar_width = 0.35
            
            # Get metric values
            training_means = []
            training_stds = []
            test_means = []
            test_stds = []
            
            for group in common_groups:
                if group in training_stats:
                    training_metric_stats = getattr(training_stats[group], metric_name)
                    training_means.append(training_metric_stats.mean)
                    training_stds.append(training_metric_stats.std)
                else:
                    training_means.append(0.0)
                    training_stds.append(0.0)
                
                if group in test_stats:
                    test_metric_stats = getattr(test_stats[group], metric_name)
                    test_means.append(test_metric_stats.mean)
                    test_stds.append(test_metric_stats.std)
                else:
                    test_means.append(0.0)
                    test_stds.append(0.0)
            
            # Create bars
            bars1 = ax.bar(x_pos - bar_width/2, training_means, bar_width,
                          yerr=training_stds, capsize=5, label='Training Frequency',
                          color=self.colors.get(metric_name, self.publication_colors['secondary']), alpha=0.7)
            
            bars2 = ax.bar(x_pos + bar_width/2, test_means, bar_width,
                          yerr=test_stds, capsize=5, label='Test Frequency',
                          color=self.stability_palette['Rare'], alpha=0.7)
            
            # Add individual datapoints using MetricStats.values
            for j, group_name in enumerate(common_groups):
                # Plot training frequency points
                if group_name in training_stats:
                    training_metric_stats = getattr(training_stats[group_name], metric_name)
                    if training_metric_stats.values and len(training_metric_stats.values) > 1:
                        jitter = np.random.normal(0, 0.02, len(training_metric_stats.values))
                        x_jittered = np.full(len(training_metric_stats.values), x_pos[j] - bar_width/2) + jitter
                        ax.scatter(x_jittered, training_metric_stats.values, 
                                 color=self.colors.get(metric_name, self.publication_colors['secondary']), 
                                 s=30, alpha=0.9, edgecolors='white', linewidth=1, zorder=3)
                
                # Plot test frequency points
                if group_name in test_stats:
                    test_metric_stats = getattr(test_stats[group_name], metric_name)
                    if test_metric_stats.values and len(test_metric_stats.values) > 1:
                        jitter = np.random.normal(0, 0.02, len(test_metric_stats.values))
                        x_jittered = np.full(len(test_metric_stats.values), x_pos[j] + bar_width/2) + jitter
                        ax.scatter(x_jittered, test_metric_stats.values, 
                                 color=self.stability_palette['Rare'], s=30, alpha=0.9, 
                                 edgecolors='white', linewidth=1, zorder=3)
            
            # Add value labels
            for bars, means in [(bars1, training_means), (bars2, test_means)]:
                for bar, mean in zip(bars, means):
                    if mean > 0:  # Only show labels for non-zero values
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{mean:.3f}', ha='center', va='bottom', 
                               fontsize=9, fontweight='bold')
            
            # Styling
            ax.set_xlabel('Stability Groups', fontsize=14, fontweight='bold')
            ax.set_ylabel(metric_label, fontsize=14, fontweight='bold')
            ax.set_title(f'{metric_label} by Stability Group\nAcross {aggregated_metrics.n_replicas} Replicas',
                        fontsize=16, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels([g.replace('Moderate', 'Transient') for g in common_groups])
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
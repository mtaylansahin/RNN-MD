"""Metrics visualization components for performance analysis."""

import sys
from pathlib import Path

# Add src to path for absolute imports
sys.path.insert(0, str(Path(__file__).parents[2]))

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple
from collections import defaultdict

from .base_plotter import BasePlotter
from ..data.data_processor import ProcessedData
from ..analytics.metrics_calculator import MetricsReport


class MetricsPlotter(BasePlotter):
    """Handles metrics visualizations for performance analysis."""
    
    def generate_plots(
        self,
        processed_data: ProcessedData,
        metrics_report: MetricsReport,
        scores_file_path: str
    ) -> List[str]:
        """Generate all metrics plots.
        
        Args:
            processed_data: Processed analysis data
            metrics_report: Calculated metrics report
            scores_file_path: Path to scores file for appending stability metrics
            
        Returns:
            List of generated plot file paths
        """
        generated_plots = []
        
        try:
            # Create evaluation series for time-based analysis
            gt_eval_series, pred_eval_series = self._create_evaluation_series(processed_data)
            
            # Plot metrics vs time
            metrics_time_path = self._plot_metrics_vs_time_from_report(
                metrics_report.metrics_over_time
            )
            generated_plots.append(metrics_time_path)
            
            # Plot cumulative error
            cumulative_error_path = self._plot_cumulative_error_from_report(
                metrics_report.cumulative_errors
            )
            generated_plots.append(cumulative_error_path)
            
            # Plot stability-based metrics if available
            if processed_data.stability_bins is not None and metrics_report.metrics_by_stability:
                stability_metrics_path = self._plot_metrics_by_stability(
                    metrics_report,
                    processed_data.stability_bins,
                    scores_file_path
                )
                generated_plots.append(stability_metrics_path)
                
                # Calculate and plot per-edge F1 scores
                f1_df = self._calculate_per_edge_f1(
                    processed_data.ground_truth_full, processed_data.predictions_full
                )
                
                if f1_df is not None:
                    f1_dist_path = self._plot_f1_distribution_by_stability(
                        f1_df, processed_data.stability_bins
                    )
                    generated_plots.append(f1_dist_path)
                    
                    # Plot training frequency vs test F1 if available
                    if processed_data.pair_freq_train is not None:
                        scatter_path = self._plot_train_freq_vs_test_f1(
                            processed_data.pair_freq_train, f1_df, processed_data.stability_bins
                        )
                        generated_plots.append(scatter_path)
            
            # Plot Mean Pairwise F1 Score comparison
            if hasattr(metrics_report, 'mean_pairwise_f1'):
                mean_f1_path = self._plot_mean_pairwise_f1_comparison(metrics_report)
                generated_plots.append(mean_f1_path)
            
            self.logger.info(f"Generated {len(generated_plots)} metrics plots")
            
        except Exception as e:
            self.logger.error(f"Failed to generate metrics plots: {e}")
        
        return generated_plots
    
    def _create_evaluation_series(self, processed_data: ProcessedData) -> Tuple[pd.Series, pd.Series]:
        """Create evaluation series from processed data.
        
        Args:
            processed_data: Processed analysis data
            
        Returns:
            Tuple of (ground_truth_series, predictions_series)
        """
        # Create evaluation series with MultiIndex (pair, time_stamp)
        gt_eval = processed_data.ground_truth_full.set_index(['pair', 'time_stamp'])['present']
        pred_eval = processed_data.predictions_full.set_index(['pair', 'time_stamp'])['present']
        
        return gt_eval, pred_eval
    
    def _calculate_metrics(self, gt_series: pd.Series, pred_series: pd.Series, total_possible: int) -> Dict[str, float]:
        """Calculate standard metrics from binary series.
        
        Args:
            gt_series: Ground truth binary series
            pred_series: Prediction binary series
            total_possible: Total possible interactions
            
        Returns:
            Dictionary of calculated metrics
        """
        TP = ((pred_series == 1) & (gt_series == 1)).sum()
        FP = ((pred_series == 1) & (gt_series == 0)).sum()
        FN = ((pred_series == 0) & (gt_series == 1)).sum()
        TN = total_possible - (TP + FP + FN)
        TN = max(0, TN)
        
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        tpr = recall
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
        f1 = 2 * ((precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0
        mcc_denom = ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))**(1/2)
        mcc = (TP * TN - FP * FN) / mcc_denom if mcc_denom > 0 else 0
        
        return {
            'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
            'Recall': recall, 'Precision': precision, 'TPR': tpr, 'FPR': fpr,
            'F1': f1, 'MCC': mcc
        }
    
    def _plot_metrics_vs_time(
        self,
        gt_eval_series: pd.Series,
        pred_eval_series: pd.Series,
        timestamps: List[int],
        total_possible_pairs_per_ts: int
    ) -> str:
        """Plot metrics cumulatively over time.
        
        DEPRECATED: This logic is now in MetricsCalculator and results are in MetricsReport.
                    Use _plot_metrics_vs_time_from_report instead.
        
        Args:
            gt_eval_series: Ground truth evaluation series
            pred_eval_series: Prediction evaluation series
            timestamps: List of timestamps
            total_possible_pairs_per_ts: Total possible pairs per timestamp
            
        Returns:
            Path to saved plot
        """
        self.logger.warning("_plot_metrics_vs_time is deprecated. Metrics should be pre-calculated.")
        # Minimal pass-through or raise error, actual plotting should use report data
        # For now, retain original logic if strictly necessary for some interim path, but flag for removal

        self.logger.info("Generating metrics vs time plot (using old calculation method)")
        
        metrics_over_time = defaultdict(list)
        timestamps_sorted = sorted(timestamps)
        
        common_index = gt_eval_series.index.intersection(pred_eval_series.index)
        gt_aligned = gt_eval_series[common_index]
        pred_aligned = pred_eval_series[common_index]
        
        times = gt_aligned.index.get_level_values(1)
        
        for t_idx, t in enumerate(timestamps_sorted):
            mask = times <= t
            gt_cumulative = gt_aligned[mask]
            pred_cumulative = pred_aligned[mask]
            
            total_possible_cumulative = total_possible_pairs_per_ts * (t_idx + 1)
            
            metrics_calc = self._calculate_metrics(gt_cumulative, pred_cumulative, total_possible_cumulative)
            
            metrics_over_time['Time'].append(t)
            for key in ['Recall', 'Precision', 'F1', 'MCC', 'TPR', 'FPR']:
                metrics_over_time[key].append(metrics_calc[key])
        
        metrics_df = pd.DataFrame(metrics_over_time)
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
        axes = axes.flatten()
        metrics_to_plot = ['Recall', 'Precision', 'F1', 'MCC', 'TPR', 'FPR']
        
        for i, metric_name in enumerate(metrics_to_plot):
            axes[i].plot(
                metrics_df['Time'], metrics_df[metric_name],
                marker='.', linestyle='-', label=metric_name
            )
            axes[i].set_title(f'Cumulative {metric_name} vs. Time')
            axes[i].set_ylabel(metric_name)
            axes[i].grid(True, linestyle='--', alpha=0.6)
            if i >= 4:  # Bottom row
                axes[i].set_xlabel('Time Stamp')
        
        plt.tight_layout()
        
        filename = 'metrics_vs_time.png'
        plot_path = self.save_plot(filename, fig)
        self.close_plot(fig)
        
        return plot_path

    def _plot_metrics_vs_time_from_report(self, metrics_over_time_df: pd.DataFrame) -> str:
        """Plot metrics cumulatively over time using data from MetricsReport."""
        self.logger.info("Generating metrics vs time plot from report data")

        if metrics_over_time_df.empty:
            self.logger.warning("Metrics over time data is empty, skipping plot.")
            return ""

        fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
        axes = axes.flatten()
        # Columns in metrics_over_time_df: Time, Recall, Precision, F1, MCC, TPR, FPR
        metrics_to_plot = ['Recall', 'Precision', 'F1', 'MCC', 'TPR', 'FPR']
        
        for i, metric_name in enumerate(metrics_to_plot):
            if metric_name not in metrics_over_time_df.columns:
                self.logger.warning(f"Metric {metric_name} not found in metrics_over_time_df. Skipping.")
                axes[i].set_title(f'Cumulative {metric_name} vs. Time (Data N/A)')
                axes[i].set_ylabel(metric_name)
                if i >= 4: axes[i].set_xlabel('Time Stamp')
                axes[i].grid(True, linestyle='--', alpha=0.6)
                continue
            
            axes[i].plot(
                metrics_over_time_df['Time'], metrics_over_time_df[metric_name],
                marker='.', linestyle='-', label=metric_name
            )
            axes[i].set_title(f'Cumulative {metric_name} vs. Time')
            axes[i].set_ylabel(metric_name)
            axes[i].grid(True, linestyle='--', alpha=0.6)
            if i >= 4:  # Bottom row
                axes[i].set_xlabel('Time Stamp')
        
        plt.tight_layout()
        filename = 'metrics_vs_time.png'
        plot_path = self.save_plot(filename, fig)
        self.close_plot(fig)
        return plot_path
    
    def _plot_cumulative_error(
        self,
        gt_eval_series: pd.Series,
        pred_eval_series: pd.Series,
        timestamps: List[int]
    ) -> str:
        """Plot cumulative errors (FP + FN) over time.

        DEPRECATED: This logic is now in MetricsCalculator and results are in MetricsReport.
                    Use _plot_cumulative_error_from_report instead.
        """
        self.logger.warning("_plot_cumulative_error is deprecated. Metrics should be pre-calculated.")
        # Minimal pass-through or raise error

        self.logger.info("Generating cumulative error plot (using old calculation method)")
        errors_over_time = defaultdict(list)
        timestamps_sorted = sorted(timestamps)
        
        common_index = gt_eval_series.index.intersection(pred_eval_series.index)
        gt_aligned = gt_eval_series[common_index]
        pred_aligned = pred_eval_series[common_index]
        
        times = gt_aligned.index.get_level_values(1)
        
        cumulative_fp = 0
        cumulative_fn = 0
        
        for t in timestamps_sorted:
            mask_t = times == t
            gt_t = gt_aligned[mask_t]
            pred_t = pred_aligned[mask_t]
            
            fp_t = ((pred_t == 1) & (gt_t == 0)).sum()
            fn_t = ((pred_t == 0) & (gt_t == 1)).sum()
            
            cumulative_fp += fp_t
            cumulative_fn += fn_t
            
            errors_over_time['Time'].append(t)
            errors_over_time['Cumulative FP'].append(cumulative_fp)
            errors_over_time['Cumulative FN'].append(cumulative_fn)
            errors_over_time['Cumulative Errors (FP+FN)'].append(cumulative_fp + cumulative_fn)
        
        errors_df = pd.DataFrame(errors_over_time)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            errors_df['Time'], errors_df['Cumulative Errors (FP+FN)'],
            marker='.', linestyle='-', label='Cumulative Errors (FP+FN)'
        )
        
        ax.set_title('Cumulative Errors (FP + FN) vs. Time')
        ax.set_xlabel('Time Stamp')
        ax.set_ylabel('Cumulative Count')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        
        filename = 'cumulative_error_vs_time.png'
        plot_path = self.save_plot(filename, fig)
        self.close_plot(fig)
        return plot_path

    def _plot_cumulative_error_from_report(self, cumulative_errors_df: pd.DataFrame) -> str:
        """Plot cumulative errors (FP + FN) over time using data from MetricsReport."""
        self.logger.info("Generating cumulative error plot from report data")

        if cumulative_errors_df.empty:
            self.logger.warning("Cumulative errors data is empty, skipping plot.")
            return ""
        
        # Expected columns: Time, Cumulative_FP, Cumulative_FN, Cumulative_Errors
        if (not 'Cumulative_Errors' in cumulative_errors_df.columns or 
            not 'Time' in cumulative_errors_df.columns):
            self.logger.error("Required columns missing in cumulative_errors_df. Skipping plot.")
            return ""

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            cumulative_errors_df['Time'], cumulative_errors_df['Cumulative_Errors'],
            marker='.', linestyle='-', label='Cumulative Errors (FP+FN)'
        )
        
        ax.set_title('Cumulative Errors (FP + FN) vs. Time')
        ax.set_xlabel('Time Stamp')
        ax.set_ylabel('Cumulative Count')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        filename = 'cumulative_error_vs_time.png'
        plot_path = self.save_plot(filename, fig)
        self.close_plot(fig)
        return plot_path
    
    def _plot_metrics_by_stability(
        self,
        metrics_report: MetricsReport,
        stability_bins_info: pd.Series,
        scores_file_path: str
    ) -> str:
        """Plot performance metrics by stability bins using data from MetricsReport."""
        self.logger.info("Generating metrics by stability plot from report data")
        
        # metrics_report.metrics_by_stability is Dict[str, StabilityBinDetail]
        # StabilityBinDetail has .pair_count and inherits .Recall, .Precision, etc.

        if not metrics_report.metrics_by_stability:
            self.logger.warning("No metrics by stability data in report, skipping plot.")
            return ""

        # Prepare data for plotting from the report
        # The dict in report is: {bin_label: StabilityBinDetail_object}
        # StabilityBinDetail_object has attributes: TP, FP, FN, TN, Recall, Precision, ..., pair_count
        plot_data = []
        for bin_label, stability_detail in metrics_report.metrics_by_stability.items():
            plot_data.append({
                'Bin Label': bin_label,
                'Recall': stability_detail.Recall,
                'Precision': stability_detail.Precision,
                'F1': stability_detail.F1,
                'MCC': stability_detail.MCC,
                'Mean Pairwise F1': stability_detail.mean_pairwise_f1,
                'Baseline Mean Pairwise F1': stability_detail.baseline_mean_pairwise_f1 if stability_detail.baseline_mean_pairwise_f1 is not None else None,
                'Pair Count': stability_detail.pair_count
            })
        
        metrics_df_from_report = pd.DataFrame(plot_data).set_index('Bin Label')
        
        # Append to scores file using report data
        with open(scores_file_path, "a") as scores_output_file:
            print("\n--- Performance by Interaction Stability (TRAINING Set Frequency) ---", file=scores_output_file)
            
            # Use the order from metrics_report.metrics_by_stability which should be consistent
            # or define a specific order if necessary (e.g. Rare, Moderate, Stable, Undefined)
            # The report might have bins like 'Undefined'. Plotting excludes 'Undefined'.
            bin_order_for_scoring = list(metrics_report.metrics_by_stability.keys())

            for bin_label in bin_order_for_scoring:
                if bin_label not in metrics_report.metrics_by_stability:
                    continue # Should not happen if iterating keys
                
                stability_detail = metrics_report.metrics_by_stability[bin_label]
                pair_count = stability_detail.pair_count
                
                output_label = bin_label
                if bin_label == "Moderate (5-50%)": output_label = "Uncommon (5-50%)"
                if bin_label == "Undefined": output_label = "Not in Train"
                
                print(f"\nMetrics for {output_label} interactions ({pair_count} pairs):", file=scores_output_file)
                
                if pair_count == 0 and stability_detail.TP == 0: # Check if truly no data
                    print("No interactions found or processed in this bin for metric calculation.", file=scores_output_file)
                else:
                    # Using a consistent set of metrics for the text file as per MetricsCalculator output
                    print(f"Recall: {stability_detail.Recall:.4f}, Precision: {stability_detail.Precision:.4f}, "
                          f"TPR: {stability_detail.TPR:.4f}, FPR: {stability_detail.FPR:.4f}, "
                          f"F1: {stability_detail.F1:.4f}, MCC: {stability_detail.MCC:.4f}, "
                          f"Mean Pairwise F1: {stability_detail.mean_pairwise_f1:.4f}", file=scores_output_file)
                    if stability_detail.baseline_mean_pairwise_f1 is not None:
                        print(f"Baseline Mean Pairwise F1: {stability_detail.baseline_mean_pairwise_f1:.4f}", file=scores_output_file)

        # Prepare DataFrame for plotting (excluding 'Undefined' bin)
        metrics_df_plot = metrics_df_from_report.drop('Undefined', errors='ignore')
        
        # Dynamically select columns based on what data is available
        columns_to_plot = ['Recall', 'Precision', 'F1', 'MCC', 'Mean Pairwise F1']
        
        # Add baseline Mean Pairwise F1 if it exists and has non-null values
        if 'Baseline Mean Pairwise F1' in metrics_df_plot.columns:
            has_baseline_data = metrics_df_plot['Baseline Mean Pairwise F1'].notna().any()
            if has_baseline_data:
                columns_to_plot.append('Baseline Mean Pairwise F1')
        
        metrics_df_plot = metrics_df_plot[columns_to_plot] # Select metrics to plot
        
        # Ensure desired plot order
        # Plot order for x-axis
        plot_order_preference = ['Rare (<5%)', 'Moderate (5-50%)', 'Stable (>50%)']
        # Filter this order to only include bins actually present in metrics_df_plot
        actual_plot_order = [l for l in plot_order_preference if l in metrics_df_plot.index]
        
        if not actual_plot_order: # if, after filtering, no standard bins are left
             if metrics_df_plot.empty:
                self.logger.warning("No data to plot for metrics by stability after filtering.")
                return ""
             else: # Plot whatever is left, not in preferred order
                actual_plot_order = metrics_df_plot.index.tolist()

        metrics_df_plot = metrics_df_plot.reindex(actual_plot_order)
        
        if not metrics_df_plot.empty:
            # Get stability color palette (This seems to be a custom method)
            # stability_colors = self.get_color_palette("stability") 
            # This was not used in the original bar plot for bar colors, but for x-tick labels potentially.
            # The original plot used specific colors per metric, not per stability bin for the bars themselves.

            # Create custom color map for the metrics (as in original code)
            metric_colors = {
                'Recall': '#2E86AB',                    # Blue
                'Precision': '#A23B72',                 # Purple  
                'F1': '#F18F01',                        # Orange
                'MCC': '#C73E1D',                       # Red
                'Mean Pairwise F1': '#36213E',          # Dark Purple
                'Baseline Mean Pairwise F1': '#7A9E7E'  # Green
            }
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create the bar plot with custom colors
            # Ensure the columns exist before trying to use them for colors
            cols_to_plot_in_bar = [col for col in columns_to_plot if col in metrics_df_plot.columns]
            colors_for_bars = [metric_colors[col] for col in cols_to_plot_in_bar]

            metrics_df_plot[cols_to_plot_in_bar].plot(
                kind='bar', 
                ax=ax, 
                color=colors_for_bars, # Use filtered list of colors
                width=0.8,
                edgecolor='white',
                linewidth=0.7
            )
            
            # Add value labels on bars
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f', label_type='edge', padding=3, fontsize=9, fontweight='bold')
            
            # Create x-axis labels with sample counts
            x_labels_with_counts = []
            for bin_label_for_plot in actual_plot_order:
                # Get pair count from the original full metrics_df_from_report
                count = metrics_df_from_report.loc[bin_label_for_plot, 'Pair Count'] if bin_label_for_plot in metrics_df_from_report.index else 0
                # Handle display name for 'Moderate (5-50%)'
                display_label = "Uncommon (5-50%)" if bin_label_for_plot == "Moderate (5-50%)" else bin_label_for_plot
                x_labels_with_counts.append(f'{display_label}\n(N={int(count)})')
            
            ax.set_xticklabels(x_labels_with_counts, rotation=0, ha='center')
            
            # Styling improvements
            ax.set_title('Performance Metrics by Interaction Stability (based on Training Freq.)', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Stability Bin (Training Set Frequency)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Score', fontsize=12, fontweight='bold')
            
            # Improve legend
            ax.legend(
                title='Metric', 
                title_fontsize=11,
                fontsize=10,
                bbox_to_anchor=(1.02, 1), 
                loc='upper left',
                frameon=True,
                fancybox=True,
                shadow=True
            )
            
            # Improve grid
            ax.grid(True, axis='y', linestyle='--', alpha=0.4, linewidth=0.8)
            ax.set_axisbelow(True)
            
            # Set y-axis limits with some padding
            ax.set_ylim(bottom=0, top=min(1.1, max(1.05, ax.get_ylim()[1] * 1.08)))
            
            # Improve tick formatting
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.tick_params(axis='x', which='major', pad=5)
            
            # Add subtle background color
            ax.set_facecolor('#f8f9fa')
            
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            
            # Save and close
            filename = 'metrics_by_stability_bar_trainfreq.png'
            plot_path = self.save_plot(filename, fig)
            self.close_plot(fig)
            
            return plot_path
        else:
            self.logger.warning("No data to plot for metrics by stability")
            return ""
    
    def _calculate_per_edge_f1(
        self,
        ground_truth_full: pd.DataFrame,
        pred_full: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        """Calculate F1 score for each edge.
        
        Args:
            ground_truth_full: Ground truth full interaction grid
            pred_full: Predictions full interaction grid
            
        Returns:
            DataFrame with F1 scores per edge or None if calculation fails
        """
        # Merge ground truth and predictions
        merged = pd.merge(
            ground_truth_full.add_suffix('_gt'),
            pred_full.add_suffix('_pred'),
            left_on=['pair_gt', 'time_stamp_gt'],
            right_on=['pair_pred', 'time_stamp_pred'],
            how='inner'
        )
        
        if merged.empty:
            self.logger.warning("Merging ground truth and predictions for per-edge F1 resulted in empty dataframe")
            return None
        
        per_edge_stats = []
        for pair, group in merged.groupby('pair_gt'):
            gt = group['present_gt']
            pred = group['present_pred']
            
            TP = ((pred == 1) & (gt == 1)).sum()
            FP = ((pred == 1) & (gt == 0)).sum()
            FN = ((pred == 0) & (gt == 1)).sum()
            
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            per_edge_stats.append({'pair': pair, 'F1': f1})
        
        if not per_edge_stats:
            self.logger.warning("Could not calculate per-edge F1 stats")
            return None
        
        f1_df = pd.DataFrame(per_edge_stats).set_index('pair')
        return f1_df
    
    def _plot_f1_distribution_by_stability(
        self,
        f1_df: pd.DataFrame,
        stability_bins: pd.Series
    ) -> str:
        """Plot F1 score distribution by stability bins.
        
        Args:
            f1_df: DataFrame with F1 scores per edge
            stability_bins: Stability bin assignments
            
        Returns:
            Path to saved plot
        """
        self.logger.info("Generating F1 distribution by stability plot")
        
        # Add stability bin information
        eval_pairs_f1 = f1_df.index
        stability_bins_aligned_f1 = stability_bins.reindex(eval_pairs_f1).cat.add_categories('Undefined').fillna('Undefined')
        
        f1_with_bins = f1_df.join(stability_bins_aligned_f1.rename('Stability Bin'))
        f1_df_plot = f1_with_bins[f1_with_bins['Stability Bin'] != 'Undefined']
        
        if f1_df_plot.empty:
            self.logger.warning("No data to plot for F1 distribution by stability")
            return ""
        
        # Create plot
        plt.figure(figsize=(10, 7))
        bin_order = [b for b in ['Rare (<5%)', 'Moderate (5-50%)', 'Stable (>50%)'] if b in f1_df_plot['Stability Bin'].unique()]
        
        if bin_order:
            sns.boxplot(data=f1_df_plot, x='Stability Bin', y='F1', order=bin_order, palette='viridis')
            plt.title('Distribution of Per-Pair F1 Scores by Stability Bin (based on Training Freq.)')
            plt.xlabel('Stability Bin (Training Set Frequency)')
            plt.ylabel('F1 Score (calculated on Test Set)')
            plt.grid(True, axis='y', linestyle='--', alpha=0.6)
            plt.tight_layout()
            
            # Save and close
            filename = 'per_edge_f1_distribution_trainfreq.png'
            plot_path = self.save_plot(filename)
            plt.close()
            
            return plot_path
        else:
            self.logger.warning("No valid stability bins found for plotting F1 distribution")
            return ""
    
    def _plot_train_freq_vs_test_f1(
        self,
        pair_freq_train: pd.Series,
        f1_df: pd.DataFrame,
        stability_bins: pd.Series
    ) -> str:
        """Plot training frequency vs test F1 score scatter plot.
        
        Args:
            pair_freq_train: Training frequency data
            f1_df: DataFrame with F1 scores per edge
            stability_bins: Stability bin assignments
            
        Returns:
            Path to saved plot
        """
        self.logger.info("Generating training frequency vs test F1 scatter plot")
        
        # Combine the data
        combined_df = pd.DataFrame({
            'Train Frequency': pair_freq_train,
            'Stability Bin': stability_bins
        })
        
        # Join with Test F1 scores
        combined_df = combined_df.join(f1_df, how='inner')
        
        if combined_df.empty:
            self.logger.warning("No common pairs found between training frequency and test F1 results")
            return ""
        
        # Drop rows with missing data
        combined_df = combined_df.dropna(subset=['Stability Bin', 'Train Frequency', 'F1'])
        
        plt.figure(figsize=(12, 8))
        bin_order = [b for b in ['Rare (<5%)', 'Moderate (5-50%)', 'Stable (>50%)'] if b in combined_df['Stability Bin'].unique()]
        
        sns.scatterplot(
            data=combined_df,
            x='Train Frequency',
            y='F1',
            hue='Stability Bin',
            hue_order=bin_order,
            palette='viridis',
            alpha=0.8,
            s=60
        )
        
        plt.title('Training Set Frequency vs. Test Set F1 Score per Pair')
        plt.xlabel('Pair Frequency in Training Set')
        plt.ylabel('F1 Score on Test Set')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(title='Stability Bin (Train Freq.)')
        plt.ylim(-0.05, 1.05)
        plt.xlim(-0.05, 1.05)
        
        plt.tight_layout()
        
        # Save and close
        filename = 'scatter_train_freq_vs_test_f1.png'
        plot_path = self.save_plot(filename)
        plt.close()
        
        return plot_path
    
    def _plot_mean_pairwise_f1_comparison(self, metrics_report: MetricsReport) -> str:
        """Plot Mean Pairwise F1 Score compared with other metrics.
        
        Args:
            metrics_report: Complete metrics report
            
        Returns:
            Path to saved plot
        """
        self.logger.info("Generating Mean Pairwise F1 Score comparison plot")
        
        try:
            # Prepare data for comparison
            metrics_data = {
                'Overall F1': metrics_report.model_metrics.F1,
                'Mean Pairwise F1': metrics_report.mean_pairwise_f1,
                'Precision': metrics_report.model_metrics.Precision,
                'Recall': metrics_report.model_metrics.Recall,
                'MCC': metrics_report.model_metrics.MCC
            }
            
            # Add baseline metrics if available
            if metrics_report.baseline_metrics and metrics_report.baseline_mean_pairwise_f1 is not None:
                baseline_data = {
                    'Baseline Overall F1': metrics_report.baseline_metrics.F1,
                    'Baseline Mean Pairwise F1': metrics_report.baseline_mean_pairwise_f1,
                    'Baseline Precision': metrics_report.baseline_metrics.Precision,
                    'Baseline Recall': metrics_report.baseline_metrics.Recall,
                    'Baseline MCC': metrics_report.baseline_metrics.MCC
                }
                metrics_data.update(baseline_data)
            
            # Adjust figure size based on number of metrics
            fig_width = max(15, len(metrics_data) * 1.5)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, 6))
            
            # Left plot: Bar chart comparing all metrics
            metrics_names = list(metrics_data.keys())
            metrics_values = list(metrics_data.values())
            
            # Color palette for different metrics (extend for baseline)
            colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#36213E', '#7A9E7E', '#D4A574', '#E07A5F', '#81B29A', '#F2CC8F']
            
            bars = ax1.bar(metrics_names, metrics_values, color=colors, alpha=0.8, edgecolor='white', linewidth=1)
            
            # Add value labels on bars
            for bar, value in zip(bars, metrics_values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            ax1.set_title('Mean Pairwise F1 vs Other Metrics', fontsize=14, fontweight='bold', pad=20)
            ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
            ax1.set_ylim(0, min(1.1, max(metrics_values) * 1.15))
            ax1.grid(True, axis='y', linestyle='--', alpha=0.3)
            ax1.tick_params(axis='x', rotation=45, labelsize=10)
            
            # Right plot: Highlight Mean Pairwise F1 specifically
            f1_comparison = {
                'Overall F1': metrics_report.model_metrics.F1,
                'Mean Pairwise F1': metrics_report.mean_pairwise_f1
            }
            
            # Add baseline F1 comparisons if available
            if metrics_report.baseline_metrics and metrics_report.baseline_mean_pairwise_f1 is not None:
                f1_comparison.update({
                    'Baseline Overall F1': metrics_report.baseline_metrics.F1,
                    'Baseline Mean Pairwise F1': metrics_report.baseline_mean_pairwise_f1
                })
            
            f1_names = list(f1_comparison.keys())
            f1_values = list(f1_comparison.values())
            f1_colors = ['#2E86AB', '#A23B72', '#7A9E7E', '#D4A574'][:len(f1_names)]
            
            bars2 = ax2.bar(f1_names, f1_values, color=f1_colors, alpha=0.8, edgecolor='white', linewidth=1)
            
            # Add value labels on bars
            for bar, value in zip(bars2, f1_values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
            
            ax2.set_title('F1 Score Comparison', fontsize=14, fontweight='bold', pad=20)
            ax2.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
            ax2.set_ylim(0, min(1.1, max(f1_values) * 1.15))
            ax2.grid(True, axis='y', linestyle='--', alpha=0.3)
            
            # Add explanation text
            explanation = ("Mean Pairwise F1: Average of F1 scores calculated\nfor each interaction pair across time")
            fig.suptitle(explanation, fontsize=10, style='italic', y=0.02)
            
            plt.tight_layout(rect=[0, 0.08, 1, 1])
            
            # Save and close
            filename = 'mean_pairwise_f1_comparison.png'
            plot_path = self.save_plot(filename, fig)
            self.close_plot(fig)
            
            return plot_path
            
        except Exception as e:
            self.logger.error(f"Failed to create Mean Pairwise F1 comparison plot: {e}")
            return "" 
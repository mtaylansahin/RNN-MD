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
            metrics_time_path = self._plot_metrics_vs_time(
                gt_eval_series, pred_eval_series, processed_data.test_timestamps,
                processed_data.total_possible_pairs
            )
            generated_plots.append(metrics_time_path)
            
            # Plot cumulative error
            cumulative_error_path = self._plot_cumulative_error(
                gt_eval_series, pred_eval_series, processed_data.test_timestamps
            )
            generated_plots.append(cumulative_error_path)
            
            # Plot stability-based metrics if available
            if processed_data.stability_bins is not None:
                stability_metrics_path = self._plot_metrics_by_stability(
                    gt_eval_series, pred_eval_series, processed_data.stability_bins,
                    processed_data.total_possible_pairs, scores_file_path
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
        
        Args:
            gt_eval_series: Ground truth evaluation series
            pred_eval_series: Prediction evaluation series
            timestamps: List of timestamps
            total_possible_pairs_per_ts: Total possible pairs per timestamp
            
        Returns:
            Path to saved plot
        """
        self.logger.info("Generating metrics vs time plot")
        
        metrics_over_time = defaultdict(list)
        timestamps_sorted = sorted(timestamps)
        
        # Align indices
        common_index = gt_eval_series.index.intersection(pred_eval_series.index)
        gt_aligned = gt_eval_series[common_index]
        pred_aligned = pred_eval_series[common_index]
        
        # Get time levels for filtering
        times = gt_aligned.index.get_level_values(1)
        
        for t_idx, t in enumerate(timestamps_sorted):
            # Filter data up to current time t
            mask = times <= t
            gt_cumulative = gt_aligned[mask]
            pred_cumulative = pred_aligned[mask]
            
            # Calculate total possible interactions up to this time
            total_possible_cumulative = total_possible_pairs_per_ts * (t_idx + 1)
            
            # Calculate metrics
            metrics = self._calculate_metrics(gt_cumulative, pred_cumulative, total_possible_cumulative)
            
            # Store metrics
            metrics_over_time['Time'].append(t)
            for key in ['Recall', 'Precision', 'F1', 'MCC', 'TPR', 'FPR']:
                metrics_over_time[key].append(metrics[key])
        
        metrics_df = pd.DataFrame(metrics_over_time)
        
        # Create plot
        fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
        axes = axes.flatten()
        metrics_to_plot = ['Recall', 'Precision', 'F1', 'MCC', 'TPR', 'FPR']
        
        for i, metric in enumerate(metrics_to_plot):
            axes[i].plot(
                metrics_df['Time'], metrics_df[metric],
                marker='.', linestyle='-', label=metric
            )
            axes[i].set_title(f'Cumulative {metric} vs. Time')
            axes[i].set_ylabel(metric)
            axes[i].grid(True, linestyle='--', alpha=0.6)
            if i >= 4:  # Bottom row
                axes[i].set_xlabel('Time Stamp')
        
        plt.tight_layout()
        
        # Save and close
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
        
        Args:
            gt_eval_series: Ground truth evaluation series
            pred_eval_series: Prediction evaluation series
            timestamps: List of timestamps
            
        Returns:
            Path to saved plot
        """
        self.logger.info("Generating cumulative error plot")
        
        errors_over_time = defaultdict(list)
        timestamps_sorted = sorted(timestamps)
        
        # Align indices
        common_index = gt_eval_series.index.intersection(pred_eval_series.index)
        gt_aligned = gt_eval_series[common_index]
        pred_aligned = pred_eval_series[common_index]
        
        # Get time levels for filtering
        times = gt_aligned.index.get_level_values(1)
        
        cumulative_fp = 0
        cumulative_fn = 0
        
        for t in timestamps_sorted:
            # Filter data at current time t
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
        
        # Create plot
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
        
        # Save and close
        filename = 'cumulative_error_vs_time.png'
        plot_path = self.save_plot(filename, fig)
        self.close_plot(fig)
        
        return plot_path
    
    def _plot_metrics_by_stability(
        self,
        gt_eval_series: pd.Series,
        pred_eval_series: pd.Series,
        stability_bins: pd.Series,
        total_possible_pairs_per_ts: int,
        scores_file_path: str
    ) -> str:
        """Plot performance metrics by stability bins.
        
        Args:
            gt_eval_series: Ground truth evaluation series
            pred_eval_series: Prediction evaluation series
            stability_bins: Stability bin assignments
            total_possible_pairs_per_ts: Total possible pairs per timestamp
            scores_file_path: Path to scores file for appending metrics
            
        Returns:
            Path to saved plot
        """
        self.logger.info("Generating metrics by stability plot")
        
        metrics_by_bin = {}
        
        # Align indices
        common_index = gt_eval_series.index.intersection(pred_eval_series.index)
        gt_aligned = gt_eval_series[common_index]
        pred_aligned = pred_eval_series[common_index]
        
        # Get evaluation pairs and align with stability bins
        eval_pairs = gt_aligned.index.get_level_values(0).unique()
        stability_bins_aligned = stability_bins.reindex(eval_pairs).cat.add_categories('Undefined').fillna('Undefined')
        bin_counts = stability_bins_aligned.value_counts()
        
        # Calculate total possible interactions over time
        total_possible_interactions_over_time = total_possible_pairs_per_ts * len(gt_aligned.index.get_level_values(1).unique())
        
        # Append to scores file
        with open(scores_file_path, "a") as scores:
            print("\n--- Performance by Interaction Stability (TRAINING Set Frequency) ---", file=scores)
            
            for bin_label in stability_bins_aligned.cat.categories:
                pairs_in_bin = stability_bins_aligned[stability_bins_aligned == bin_label].index
                pair_count = bin_counts.get(bin_label, 0)
                
                # Rename for output
                output_label = "Uncommon (10-50%)" if bin_label == "Moderate (10-50%)" else bin_label
                output_label = "Not in Train" if bin_label == "Undefined" else output_label
                print(f"\nMetrics for {output_label} interactions ({pair_count} pairs):", file=scores)
                
                if pairs_in_bin.empty or pair_count == 0:
                    metrics_by_bin[bin_label] = {k: 0 for k in ['Recall', 'Precision', 'F1', 'MCC']}
                    print("No interactions found in this bin for metric calculation.", file=scores)
                    continue
                
                # Filter evaluation series to include only pairs in current bin
                gt_bin = gt_aligned[gt_aligned.index.get_level_values(0).isin(pairs_in_bin)]
                pred_bin = pred_aligned[pred_aligned.index.get_level_values(0).isin(pairs_in_bin)]
                
                if gt_bin.empty and pred_bin.empty:
                    metrics = {'Recall': 0, 'Precision': 0, 'F1': 0, 'MCC': 0}
                    print("No interactions present in test/predictions for this bin.", file=scores)
                else:
                    # Calculate metrics for this bin
                    TP_bin = ((pred_bin == 1) & (gt_bin == 1)).sum()
                    FP_bin = ((pred_bin == 1) & (gt_bin == 0)).sum()
                    FN_bin = ((pred_bin == 0) & (gt_bin == 1)).sum()
                    TN_bin = ((pred_bin == 0) & (gt_bin == 0)).sum()
                    
                    recall_bin = TP_bin / (TP_bin + FN_bin) if (TP_bin + FN_bin) > 0 else 0
                    precision_bin = TP_bin / (TP_bin + FP_bin) if (TP_bin + FP_bin) > 0 else 0
                    f1_bin = 2 * ((precision_bin * recall_bin) / (precision_bin + recall_bin)) if (precision_bin + recall_bin) > 0 else 0
                    mcc_denom_bin = ((TP_bin + FP_bin) * (TP_bin + FN_bin) * (TN_bin + FP_bin) * (TN_bin + FN_bin))**(1/2)
                    mcc_bin = (TP_bin * TN_bin - FP_bin * FN_bin) / mcc_denom_bin if mcc_denom_bin > 0 else 0
                    
                    metrics = {'Recall': recall_bin, 'Precision': precision_bin, 'F1': f1_bin, 'MCC': mcc_bin}
                    print(f"Recall: {recall_bin:.4f}, Precision: {precision_bin:.4f}, F1: {f1_bin:.4f}, MCC: {mcc_bin:.4f}", file=scores)
                
                metrics_by_bin[bin_label] = metrics
        
        # Prepare DataFrame for plotting
        metrics_df = pd.DataFrame(metrics_by_bin).T
        metrics_df_plot = metrics_df.drop('Undefined', errors='ignore')
        
        # Ensure desired plot order
        plot_order = [l for l in ['Rare (<10%)', 'Moderate (10-50%)', 'Stable (>50%)'] if l in metrics_df_plot.index]
        metrics_df_plot = metrics_df_plot.reindex(plot_order)
        
        if not metrics_df_plot.empty:
            fig, ax = plt.subplots(figsize=(12, 7))
            metrics_df_plot.plot(kind='bar', ax=ax)
            
            # Add value labels on bars
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f', label_type='edge', padding=3, fontsize=9)
            
            ax.set_title('Performance Metrics by Interaction Stability (based on Training Freq.)')
            ax.set_xlabel('Stability Bin (Training Set Frequency)')
            ax.set_ylabel('Score')
            ax.tick_params(axis='x', rotation=0)
            ax.legend(title='Metric', bbox_to_anchor=(1.02, 1), loc='upper left')
            ax.grid(True, axis='y', linestyle='--', alpha=0.6)
            ax.set_ylim(bottom=0, top=max(1.05, ax.get_ylim()[1] * 1.05))
            
            plt.tight_layout(rect=[0, 0, 0.88, 1])
            
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
        bin_order = [b for b in ['Rare (<10%)', 'Moderate (10-50%)', 'Stable (>50%)'] if b in f1_df_plot['Stability Bin'].unique()]
        
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
        bin_order = [b for b in ['Rare (<10%)', 'Moderate (10-50%)', 'Stable (>50%)'] if b in combined_df['Stability Bin'].unique()]
        
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
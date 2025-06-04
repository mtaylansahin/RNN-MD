"""Trajectory visualization components for interaction analysis."""

import sys
from pathlib import Path

# Add src to path for absolute imports
sys.path.insert(0, str(Path(__file__).parents[2]))

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any

from .base_plotter import BasePlotter
from ..data.data_processor import ProcessedData


class TrajectoryPlotter(BasePlotter):
    """Handles trajectory visualizations for interaction analysis."""
    
    def generate_plots(
        self,
        processed_data: ProcessedData,
        num_representative_pairs: int = 7
    ) -> List[str]:
        """Generate all trajectory plots.
        
        Args:
            processed_data: Processed analysis data
            num_representative_pairs: Number of representative pairs to plot
            
        Returns:
            List of generated plot file paths
        """
        generated_plots = []
        
        try:
            # Select representative pairs based on test frequency
            test_pairs = self._select_representative_pairs(
                processed_data.ground_truth_full, num_representative_pairs
            )
            
            if test_pairs:
                # Plot trajectories selected by test frequency
                test_traj_path = self._plot_sample_trajectories(
                    processed_data.ground_truth_full,
                    processed_data.predictions_full,
                    processed_data.valid_set_processed,
                    test_pairs,
                    "test_freq"
                )
                generated_plots.append(test_traj_path)
            
            # Select representative pairs based on training frequency if available
            if processed_data.pair_freq_train is not None:
                train_pairs = self._select_representative_pairs_train_freq(
                    processed_data.train_set_processed, num_representative_pairs
                )
                
                if train_pairs:
                    # Plot trajectories selected by training frequency
                    train_traj_path = self._plot_sample_trajectories_train_selection(
                        processed_data.ground_truth_full,
                        processed_data.predictions_full,
                        processed_data.valid_set_processed,
                        train_pairs,
                        processed_data.pair_freq_train
                    )
                    generated_plots.append(train_traj_path)
            
            # Plot flip rate comparison
            flip_rate_path = self._plot_flip_rate_comparison(
                processed_data.ground_truth_full,
                processed_data.predictions_full,
                processed_data.test_timestamps
            )
            generated_plots.append(flip_rate_path)
            
            self.logger.info(f"Generated {len(generated_plots)} trajectory plots")
            
        except Exception as e:
            self.logger.error(f"Failed to generate trajectory plots: {e}")
        
        return generated_plots
    
    def _select_representative_pairs(self, gt_full: pd.DataFrame, n: int = 7) -> List[str]:
        """Select representative pairs based on test set persistence quantiles.
        
        Args:
            gt_full: Ground truth full interaction grid
            n: Number of pairs to select
            
        Returns:
            List of selected pair names
        """
        total_test_timestamps = gt_full['time_stamp'].nunique()
        if total_test_timestamps == 0:
            self.logger.warning("No timestamps found in test set ground truth")
            return []
        
        pair_persistence = gt_full[gt_full['present'] == 1].groupby('pair')['present'].count() / total_test_timestamps
        pair_persistence = pair_persistence.sort_values()
        
        if pair_persistence.empty:
            self.logger.warning("No persistent pairs found in ground truth")
            return []
        
        num_pairs_available = len(pair_persistence)
        self.logger.info(f"Found {num_pairs_available} unique pairs with interactions in ground truth")
        
        if num_pairs_available <= n:
            return pair_persistence.index.tolist()
        
        # Define quantiles for selection
        quantiles = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] if n == 7 else np.linspace(0, 1, n)
        
        selected_indices = set()
        quantile_values = pair_persistence.quantile(quantiles, interpolation='nearest')
        
        for q_val in quantile_values:
            closest_idx = (pair_persistence - q_val).abs().idxmin()
            selected_indices.add(closest_idx)
        
        # Add more pairs if duplicates were picked
        additional_needed = n - len(selected_indices)
        if additional_needed > 0:
            available_indices = pair_persistence.index.difference(list(selected_indices))
            if len(available_indices) >= additional_needed:
                additional_pairs = np.random.choice(available_indices, additional_needed, replace=False)
                selected_indices.update(additional_pairs)
            else:
                selected_indices.update(available_indices)
        
        final_selection = list(selected_indices)[:n]
        self.logger.info(f"Selected {len(final_selection)} representative pairs based on test persistence")
        return final_selection
    
    def _select_representative_pairs_train_freq(self, train_set_post: pd.DataFrame, n: int = 7) -> List[str]:
        """Select representative pairs based on training set persistence quantiles.
        
        Args:
            train_set_post: Processed training set
            n: Number of pairs to select
            
        Returns:
            List of selected pair names
        """
        if train_set_post.empty:
            self.logger.warning("Training set data is empty")
            return []
        
        total_train_timestamps = train_set_post['time_stamp'].nunique()
        if total_train_timestamps == 0:
            self.logger.warning("No timestamps found in training set")
            return []
        
        pair_counts_train = train_set_post.groupby('pair').size()
        pair_persistence_train = pair_counts_train / total_train_timestamps
        pair_persistence_train = pair_persistence_train.sort_values()
        
        if pair_persistence_train.empty:
            self.logger.warning("No pairs found in training set")
            return []
        
        num_pairs_available = len(pair_persistence_train)
        self.logger.info(f"Found {num_pairs_available} unique pairs in training set")
        
        if num_pairs_available <= n:
            return pair_persistence_train.index.tolist()
        
        # Define quantiles for selection
        quantiles = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] if n == 7 else np.linspace(0, 1, n)
        
        selected_indices = set()
        quantile_values = pair_persistence_train.quantile(quantiles, interpolation='nearest')
        
        for q_val in quantile_values:
            closest_idx = (pair_persistence_train - q_val).abs().idxmin()
            selected_indices.add(closest_idx)
        
        # Add more pairs if duplicates were picked
        additional_needed = n - len(selected_indices)
        if additional_needed > 0:
            available_indices = pair_persistence_train.index.difference(list(selected_indices))
            if len(available_indices) >= additional_needed:
                additional_pairs = np.random.choice(available_indices, additional_needed, replace=False)
                selected_indices.update(additional_pairs)
            else:
                selected_indices.update(available_indices)
        
        final_selection = list(selected_indices)[:n]
        self.logger.info(f"Selected {len(final_selection)} representative pairs based on training persistence")
        return final_selection
    
    def _plot_sample_trajectories(
        self,
        gt_full: pd.DataFrame,
        pred_full: pd.DataFrame,
        valid_set_post: pd.DataFrame,
        pairs: List[str],
        selection_type: str
    ) -> str:
        """Plot sample trajectories for selected pairs.
        
        Args:
            gt_full: Ground truth full interaction grid
            pred_full: Predictions full interaction grid
            valid_set_post: Validation set data
            pairs: List of pairs to plot
            selection_type: Type of selection used ("test_freq" or "train_freq")
            
        Returns:
            Path to saved plot
        """
        if not pairs:
            self.logger.warning(f"No pairs selected for trajectory plot ({selection_type})")
            return ""
        
        num_pairs = len(pairs)
        fig_height = max(6, 2.5 * num_pairs)
        fig, axes = plt.subplots(num_pairs, 1, figsize=(14, fig_height), sharex=True, squeeze=False)
        axes = axes.flatten()
        
        # Determine validation cutoff
        last_valid_time = -1
        if not valid_set_post.empty:
            last_valid_time = valid_set_post['time_stamp'].max()
        
        colors = self.get_color_palette()
        
        for i, pair in enumerate(pairs):
            # Get validation data for the pair
            valid_pair_gt = pd.DataFrame()
            if not valid_set_post.empty:
                valid_pair_gt = valid_set_post[valid_set_post['pair'] == pair].sort_values('time_stamp')
                if not valid_pair_gt.empty and 'present' not in valid_pair_gt.columns:
                    valid_pair_gt = valid_pair_gt.assign(present=1)
            
            # Get test data for the pair
            gt_pair_test = gt_full[gt_full['pair'] == pair].sort_values('time_stamp')
            pred_pair_test = pred_full[pred_full['pair'] == pair].sort_values('time_stamp')
            
            # Combine validation and test ground truth
            combined_gt_list = []
            if not valid_pair_gt.empty:
                combined_gt_list.append(valid_pair_gt[['time_stamp', 'present']])
            if not gt_pair_test.empty:
                combined_gt_list.append(gt_pair_test[['time_stamp', 'present']])
            
            combined_gt_df = pd.DataFrame()
            if combined_gt_list:
                combined_gt_df = pd.concat(combined_gt_list).drop_duplicates(
                    subset=['time_stamp'], keep='first'
                ).sort_values('time_stamp')
            
            # Prepare prediction series
            combined_pred_list = []
            if not valid_pair_gt.empty:
                combined_pred_list.append(valid_pair_gt[['time_stamp', 'present']])
            if not pred_pair_test.empty:
                combined_pred_list.append(pred_pair_test[['time_stamp', 'present']])
            
            combined_pred_df = pd.DataFrame()
            if combined_pred_list:
                combined_pred_df = pd.concat(combined_pred_list).drop_duplicates(
                    subset=['time_stamp'], keep='first'
                ).sort_values('time_stamp')
            
            # Determine time range
            all_times = set()
            if not combined_gt_df.empty:
                all_times.update(combined_gt_df['time_stamp'])
            if not combined_pred_df.empty:
                all_times.update(combined_pred_df['time_stamp'])
            
            if not all_times:
                axes[i].set_title(f'Pair: {pair} (No data available)')
                axes[i].axis('off')
                continue
            
            time_range_sorted = sorted(list(all_times))
            
            # Reindex to full time range
            gt_plot = pd.Series(index=time_range_sorted, dtype='float64')
            if not combined_gt_df.empty:
                gt_plot = combined_gt_df.set_index('time_stamp')['present'].reindex(
                    time_range_sorted, fill_value=0
                )
            
            pred_plot = pd.Series(index=time_range_sorted, dtype='float64')
            if not combined_pred_df.empty:
                pred_plot = combined_pred_df.set_index('time_stamp')['present'].reindex(
                    time_range_sorted, fill_value=0
                )
            
            # Plot trajectories
            axes[i].step(
                gt_plot.index, gt_plot.values, where='post',
                label='Ground Truth (Valid+Test)', color=colors["ground_truth"], linewidth=1.5
            )
            axes[i].step(
                pred_plot.index, pred_plot.values + 0.05, where='post',
                label='Prediction (Valid GT+Test Pred)', color=colors["prediction"],
                linestyle='--', linewidth=1.5
            )
            
            # Add validation/test separator
            if last_valid_time != -1 and last_valid_time < time_range_sorted[-1]:
                axes[i].axvline(
                    x=last_valid_time + 0.5, color='red', linestyle='--',
                    linewidth=1.2, label='Valid/Test Cutoff'
                )
                
                # Add period labels
                min_plot_time, max_plot_time = time_range_sorted[0], time_range_sorted[-1]
                if last_valid_time >= min_plot_time:
                    axes[i].text(
                        (min_plot_time + last_valid_time) / 2, 1.08, 'Validation',
                        ha='center', va='bottom', color='red', fontsize=9
                    )
                if last_valid_time < max_plot_time:
                    axes[i].text(
                        (last_valid_time + 1 + max_plot_time) / 2, 1.08, 'Test',
                        ha='center', va='bottom', color='black', fontsize=9
                    )
            
            # Calculate persistence for title
            gt_persistence_test = gt_pair_test['present'].mean() if not gt_pair_test.empty else 0
            axes[i].set_title(f'Pair: {pair} (Test Persistence: {gt_persistence_test:.2f})')
            axes[i].set_yticks([0, 1])
            axes[i].set_yticklabels(['Off', 'On'])
            axes[i].set_ylim(-0.1, 1.15)
            axes[i].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
            axes[i].grid(True, axis='y', linestyle=':', alpha=0.7)
            
            if i == num_pairs - 1:
                axes[i].set_xlabel('Time Stamp (Validation + Test)')
            else:
                axes[i].tick_params(axis='x', labelbottom=False)
        
        title_suffix = "Test Freq." if selection_type == "test_freq" else "Train Freq."
        fig.suptitle(
            f'Sample Pair Trajectories: Validation Ground Truth + Test Performance (Pairs Selected by {title_suffix})',
            fontsize=14, y=0.99
        )
        plt.tight_layout(rect=[0, 0.03, 0.9, 0.97])
        
        # Save and close
        filename = f'sample_pair_trajectories_{selection_type}_with_validation.png'
        plot_path = self.save_plot(filename, fig)
        self.close_plot(fig)
        
        return plot_path
    
    def _plot_sample_trajectories_train_selection(
        self,
        gt_full: pd.DataFrame,
        pred_full: pd.DataFrame,
        valid_set_post: pd.DataFrame,
        pairs_selected_by_train: List[str],
        pair_freq_train: pd.Series
    ) -> str:
        """Plot trajectories for pairs selected by training frequency.
        
        Args:
            gt_full: Ground truth full interaction grid
            pred_full: Predictions full interaction grid
            valid_set_post: Validation set data
            pairs_selected_by_train: Pairs selected by training frequency
            pair_freq_train: Training frequency data
            
        Returns:
            Path to saved plot
        """
        if not pairs_selected_by_train:
            self.logger.warning("No pairs selected by training frequency")
            return ""
        
        num_pairs = len(pairs_selected_by_train)
        fig_height = max(6, 2.5 * num_pairs)
        fig, axes = plt.subplots(num_pairs, 1, figsize=(14, fig_height), sharex=True, squeeze=False)
        axes = axes.flatten()
        
        last_valid_time = -1
        if not valid_set_post.empty:
            last_valid_time = valid_set_post['time_stamp'].max()
        
        colors = self.get_color_palette()
        
        for i, pair in enumerate(pairs_selected_by_train):
            # Get validation data
            valid_pair_gt = pd.DataFrame()
            if not valid_set_post.empty:
                valid_pair_gt = valid_set_post[valid_set_post['pair'] == pair].sort_values('time_stamp')
                if not valid_pair_gt.empty and 'present' not in valid_pair_gt.columns:
                    valid_pair_gt = valid_pair_gt.assign(present=1)
            
            # Get test data
            gt_pair_test = gt_full[gt_full['pair'] == pair].sort_values('time_stamp')
            pred_pair_test = pred_full[pred_full['pair'] == pair].sort_values('time_stamp')
            
            # Combine data (similar to previous method)
            combined_gt_list = []
            if not valid_pair_gt.empty:
                combined_gt_list.append(valid_pair_gt[['time_stamp', 'present']])
            if not gt_pair_test.empty:
                combined_gt_list.append(gt_pair_test[['time_stamp', 'present']])
            
            combined_gt_df = pd.DataFrame()
            if combined_gt_list:
                combined_gt_df = pd.concat(combined_gt_list).drop_duplicates(
                    subset=['time_stamp'], keep='first'
                ).sort_values('time_stamp')
            
            combined_pred_list = []
            if not valid_pair_gt.empty:
                combined_pred_list.append(valid_pair_gt[['time_stamp', 'present']])
            if not pred_pair_test.empty:
                combined_pred_list.append(pred_pair_test[['time_stamp', 'present']])
            
            combined_pred_df = pd.DataFrame()
            if combined_pred_list:
                combined_pred_df = pd.concat(combined_pred_list).drop_duplicates(
                    subset=['time_stamp'], keep='first'
                ).sort_values('time_stamp')
            
            all_times = set()
            if not combined_gt_df.empty:
                all_times.update(combined_gt_df['time_stamp'])
            if not combined_pred_df.empty:
                all_times.update(combined_pred_df['time_stamp'])
            
            if not all_times:
                train_persistence = pair_freq_train.get(pair, 0)
                axes[i].set_title(f'Pair: {pair} (Train Persistence: {train_persistence:.2f}) (No Valid/Test Data)')
                axes[i].axis('off')
                continue
            
            time_range_sorted = sorted(list(all_times))
            
            gt_plot = pd.Series(index=time_range_sorted, dtype='float64')
            if not combined_gt_df.empty:
                gt_plot = combined_gt_df.set_index('time_stamp')['present'].reindex(
                    time_range_sorted, fill_value=0
                )
            
            pred_plot = pd.Series(index=time_range_sorted, dtype='float64')
            if not combined_pred_df.empty:
                pred_plot = combined_pred_df.set_index('time_stamp')['present'].reindex(
                    time_range_sorted, fill_value=0
                )
            
            axes[i].step(
                gt_plot.index, gt_plot.values, where='post',
                label='Ground Truth (Valid+Test)', color=colors["ground_truth"], linewidth=1.5
            )
            axes[i].step(
                pred_plot.index, pred_plot.values + 0.05, where='post',
                label='Prediction (Valid GT+Test Pred)', color=colors["prediction"],
                linestyle='--', linewidth=1.5
            )
            
            if last_valid_time != -1 and last_valid_time < time_range_sorted[-1]:
                axes[i].axvline(
                    x=last_valid_time + 0.5, color='red', linestyle='--',
                    linewidth=1.2, label='Valid/Test Cutoff'
                )
                
                min_plot_time, max_plot_time = time_range_sorted[0], time_range_sorted[-1]
                if last_valid_time >= min_plot_time:
                    axes[i].text(
                        (min_plot_time + last_valid_time) / 2, 1.08, 'Validation',
                        ha='center', va='bottom', color='red', fontsize=9
                    )
                if last_valid_time < max_plot_time:
                    axes[i].text(
                        (last_valid_time + 1 + max_plot_time) / 2, 1.08, 'Test',
                        ha='center', va='bottom', color='black', fontsize=9
                    )
            
            train_persistence = pair_freq_train.get(pair, 0)
            axes[i].set_title(f'Pair: {pair} (Train Persistence: {train_persistence:.2f})')
            axes[i].set_yticks([0, 1])
            axes[i].set_yticklabels(['Off', 'On'])
            axes[i].set_ylim(-0.1, 1.15)
            axes[i].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
            axes[i].grid(True, axis='y', linestyle=':', alpha=0.7)
            
            if i == num_pairs - 1:
                axes[i].set_xlabel('Time Stamp (Validation + Test)')
            else:
                axes[i].tick_params(axis='x', labelbottom=False)
        
        fig.suptitle(
            'Sample Pair Trajectories: Validation Ground Truth + Test Performance (Pairs Selected by Train Freq.)',
            fontsize=14, y=0.99
        )
        plt.tight_layout(rect=[0, 0.03, 0.9, 0.97])
        
        # Save and close
        filename = 'sample_pair_trajectories_train_selection_with_validation.png'
        plot_path = self.save_plot(filename, fig)
        self.close_plot(fig)
        
        return plot_path
    
    def _plot_flip_rate_comparison(
        self,
        gt_full: pd.DataFrame,
        pred_full: pd.DataFrame,
        timestamps: List[int],
        window_size: int = 10
    ) -> str:
        """Plot flip rate comparison between ground truth and predictions.
        
        Args:
            gt_full: Ground truth full interaction grid
            pred_full: Predictions full interaction grid
            timestamps: List of timestamps
            window_size: Size of time windows for binning
            
        Returns:
            Path to saved plot
        """
        def count_flips(df: pd.DataFrame) -> pd.DataFrame:
            """Count state changes for each pair."""
            df_sorted = df.sort_values(['pair', 'time_stamp'])
            df_sorted['prev_state'] = df_sorted.groupby('pair')['present'].shift(1)
            df_sorted['flip'] = (
                (df_sorted['present'] != df_sorted['prev_state']) & 
                (df_sorted['prev_state'].notna())
            )
            return df_sorted[df_sorted['flip']]
        
        gt_flips = count_flips(gt_full)
        pred_flips = count_flips(pred_full)
        
        # Bin flips into time windows
        bins = np.arange(min(timestamps), max(timestamps) + window_size, window_size)
        labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-1)]
        
        if not labels:
            self.logger.warning("Not enough timestamps to create windows for flip rate analysis")
            return ""
        
        gt_flips['time_window'] = pd.cut(gt_flips['time_stamp'], bins=bins, labels=labels, right=False)
        pred_flips['time_window'] = pd.cut(pred_flips['time_stamp'], bins=bins, labels=labels, right=False)
        
        gt_flip_counts = gt_flips.groupby('time_window').size()
        pred_flip_counts = pred_flips.groupby('time_window').size()
        
        # Combine counts for plotting
        flip_counts_df = pd.DataFrame({
            'Ground Truth': gt_flip_counts,
            'Prediction': pred_flip_counts
        }).fillna(0)
        
        # Plot
        colors = self.get_color_palette()
        fig, ax = plt.subplots(figsize=(12, 6))
        flip_counts_df.plot(
            kind='bar', ax=ax,
            color=[colors["ground_truth"], colors["prediction"]]
        )
        ax.set_title(f'Interaction State Flips per Time Window (Size={window_size})')
        ax.set_xlabel('Time Window')
        ax.set_ylabel('Number of Flips (On<->Off)')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        
        # Save and close
        filename = 'flip_rate_comparison.png'
        plot_path = self.save_plot(filename, fig)
        self.close_plot(fig)
        
        return plot_path 
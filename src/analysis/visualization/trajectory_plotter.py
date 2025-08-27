"""Trajectory visualization components for interaction analysis."""

import sys
from pathlib import Path

# Add src to path for absolute imports
sys.path.insert(0, str(Path(__file__).parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Optional

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
            test_persistence = self._compute_persistence(processed_data.ground_truth_full)
            test_pairs = self._select_pairs_from_persistence(test_persistence, num_representative_pairs)
            
            if test_pairs:
                # Plot trajectories selected by test frequency
                test_traj_path = self._plot_sample_trajectories(
                    gt_full=processed_data.ground_truth_full,
                    pred_full=processed_data.predictions_full,
                    valid_set_post=processed_data.valid_set_processed,
                    pairs=test_pairs,
                    selection_type="test_freq"
                )
                generated_plots.append(test_traj_path)
            
            # Select representative pairs based on training frequency if available
            if processed_data.pair_freq_train is not None:
                train_persistence = self._compute_persistence(processed_data.train_set_processed)
                train_pairs = self._select_pairs_from_persistence(train_persistence, num_representative_pairs)
                
                if train_pairs:
                    # Plot trajectories selected by training frequency
                    train_traj_path = self._plot_sample_trajectories(
                        gt_full=processed_data.ground_truth_full,
                        pred_full=processed_data.predictions_full,
                        valid_set_post=processed_data.valid_set_processed,
                        pairs=train_pairs,
                        selection_type="train_freq",
                        pair_persistence_series=processed_data.pair_freq_train
                    )
                    generated_plots.append(train_traj_path)
            
            self.logger.info(f"Generated {len(generated_plots)} trajectory plots")
            
        except Exception as e:
            self.logger.error(f"Failed to generate trajectory plots: {e}")
        
        return generated_plots
    
    def _select_pairs_from_persistence(self, persistence: pd.Series, n: int) -> List[str]:
        """Helper to select pairs at quantiles from a persistence series.
        
        Args:
            persistence: Series indexed by pair with persistence values
            n: Number of pairs to select
        """
        num_pairs_available = len(persistence)
        if num_pairs_available == 0:
            return []
        if num_pairs_available <= n:
            return persistence.index.tolist()
        quantiles = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] if n == 7 else np.linspace(0, 1, n)
        selected_indices = set()
        quantile_values = persistence.quantile(quantiles, interpolation='nearest')
        for q_val in quantile_values:
            closest_idx = (persistence - q_val).abs().idxmin()
            selected_indices.add(closest_idx)
        additional_needed = n - len(selected_indices)
        if additional_needed > 0:
            available_indices = persistence.index.difference(list(selected_indices))
            if len(available_indices) >= additional_needed:
                additional_pairs = np.random.choice(available_indices, additional_needed, replace=False)
                selected_indices.update(additional_pairs)
            else:
                selected_indices.update(available_indices)
        return list(selected_indices)[:n]
    
    def _compute_persistence(self, df: pd.DataFrame) -> pd.Series:
        """Compute per-pair persistence = (#timestamps with present==1) / (#unique timestamps)."""
        if df.empty:
            return pd.Series(dtype='float64')
        total_ts = df['time_stamp'].nunique()
        if total_ts == 0:
            return pd.Series(dtype='float64')
        if 'present' in df.columns:
            counts = df[df['present'] == 1].groupby('pair')['present'].count()
        else:
            counts = df.groupby('pair')['time_stamp'].nunique()
        persistence = (counts / total_ts).sort_values()
        return persistence
    
    def _plot_sample_trajectories(
        self,
        gt_full: pd.DataFrame,
        pred_full: pd.DataFrame,
        valid_set_post: pd.DataFrame,
        pairs: List[str],
        selection_type: str,
        pair_persistence_series: Optional[pd.Series] = None
    ) -> str:
        """Common trajectory plotting for test or train selection.
        
        Args:
            gt_full: Ground truth full interaction grid
            pred_full: Predictions full interaction grid
            valid_set_post: Validation set data
            pairs: List of pairs to plot
            selection_type: "test_freq" or "train_freq"
            pair_persistence_series: Optional series for train persistence values
        """
        if not pairs:
            self.logger.warning(f"No pairs selected for trajectory plot ({selection_type})")
            return ""
        
        num_pairs = len(pairs)
        fig_height = max(6, 2.5 * num_pairs)
        fig, axes = plt.subplots(num_pairs, 1, figsize=(14, fig_height), sharex=True, squeeze=False)
        axes = axes.flatten()
        
        last_valid_time = -1
        if not valid_set_post.empty:
            last_valid_time = valid_set_post['time_stamp'].max()
        
        colors = self.get_color_palette()
        
        for i, pair in enumerate(pairs):
            valid_pair_gt = self._get_pair_valid_gt(valid_set_post, pair)
            gt_pair_test = gt_full[gt_full['pair'] == pair].sort_values('time_stamp')
            pred_pair_test = pred_full[pred_full['pair'] == pair].sort_values('time_stamp')
            
            combined_gt_df = self._combine_time_present([valid_pair_gt, gt_pair_test])
            combined_pred_df = self._combine_time_present([valid_pair_gt, pred_pair_test])
            
            all_times = sorted(list(set(combined_gt_df['time_stamp']).union(set(combined_pred_df['time_stamp'])))) if (not combined_gt_df.empty or not combined_pred_df.empty) else []
            if not all_times:
                title = self._title_for_pair(selection_type, pair, pair_persistence_series, gt_pair_test)
                axes[i].set_title(f"{title} (No Valid/Test Data)")
                axes[i].axis('off')
                continue
            time_range_sorted = all_times
            
            gt_plot = self._reindex_present_series(combined_gt_df, time_range_sorted)
            pred_plot = self._reindex_present_series(combined_pred_df, time_range_sorted)
            
            axes[i].step(
                gt_plot.index, gt_plot.values, where='post',
                label='Ground Truth (Valid+Test)', color=colors["ground_truth"], linewidth=1.5
            )
            axes[i].step(
                pred_plot.index, pred_plot.values + 0.05, where='post',
                label='Prediction (Valid GT+Test Pred)', color=colors["prediction"], linestyle='--', linewidth=1.5
            )
            
            self._annotate_periods(axes[i], last_valid_time, time_range_sorted)
            axes[i].set_title(self._title_for_pair(selection_type, pair, pair_persistence_series, gt_pair_test))
            
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
        filename = f'sample_pair_trajectories_{selection_type}_with_validation.png'
        plot_path = self.save_plot(filename, fig)
        self.close_plot(fig)
        return plot_path

    def _get_pair_valid_gt(self, valid_set_post: pd.DataFrame, pair: str) -> pd.DataFrame:
        """Return validation GT rows for a pair with ensured 'present' column."""
        if valid_set_post.empty:
            return pd.DataFrame()
        df = valid_set_post[valid_set_post['pair'] == pair].sort_values('time_stamp')
        if not df.empty and 'present' not in df.columns:
            df = df.assign(present=1)
        return df

    def _combine_time_present(self, frames: List[pd.DataFrame]) -> pd.DataFrame:
        """Concat time/present columns from non-empty frames, drop duplicate timestamps, sort."""
        sanitized = [f[['time_stamp', 'present']] for f in frames if (f is not None and not f.empty)]
        if not sanitized:
            return pd.DataFrame()
        return pd.concat(sanitized).drop_duplicates(subset=['time_stamp'], keep='first').sort_values('time_stamp')

    def _reindex_present_series(self, df: pd.DataFrame, time_range_sorted: List[int]) -> pd.Series:
        """Build a present Series over the provided time range, filling gaps with 0."""
        if df.empty:
            return pd.Series(index=time_range_sorted, dtype='float64')
        return df.set_index('time_stamp')['present'].reindex(time_range_sorted, fill_value=0)

    def _annotate_periods(self, ax, last_valid_time: int, time_range_sorted: List[int]) -> None:
        """Draw validation/test split line and labels if applicable."""
        if last_valid_time != -1 and last_valid_time < time_range_sorted[-1]:
            ax.axvline(x=last_valid_time + 0.5, color='red', linestyle='--', linewidth=1.2, label='Valid/Test Cutoff')
            min_plot_time, max_plot_time = time_range_sorted[0], time_range_sorted[-1]
            if last_valid_time >= min_plot_time:
                ax.text((min_plot_time + last_valid_time) / 2, 1.08, 'Validation', ha='center', va='bottom', color='red', fontsize=9)
            if last_valid_time < max_plot_time:
                ax.text((last_valid_time + 1 + max_plot_time) / 2, 1.08, 'Test', ha='center', va='bottom', color='black', fontsize=9)

    def _title_for_pair(
        self,
        selection_type: str,
        pair: str,
        pair_persistence_series: Optional[pd.Series],
        gt_pair_test: pd.DataFrame
    ) -> str:
        """Generate a concise title for a pair based on selection mode."""
        if selection_type == "train_freq" and pair_persistence_series is not None:
            train_persistence = float(pair_persistence_series.get(pair, 0))
            return f'Pair: {pair} (Train Persistence: {train_persistence:.2f})'
        gt_persistence_test = gt_pair_test['present'].mean() if not gt_pair_test.empty else 0
        return f'Pair: {pair} (Test Persistence: {gt_persistence_test:.2f})'
    
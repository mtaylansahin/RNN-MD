"""Heatmap visualization components for interaction analysis."""

import sys
from pathlib import Path

# Add src to path for absolute imports
sys.path.insert(0, str(Path(__file__).parents[2]))

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any

from .base_plotter import BasePlotter
from ..data.data_processor import ProcessedData


class HeatmapPlotter(BasePlotter):
    """Handles heatmap visualizations for interaction analysis."""
    
    def generate_plots(
        self,
        processed_data: ProcessedData,
        num_pairs_to_show: int = 50,
        valid_steps_to_show: int = 20
    ) -> List[str]:
        """Generate all heatmap plots.
        
        Args:
            processed_data: Processed analysis data
            num_pairs_to_show: Number of pairs to show in heatmaps
            valid_steps_to_show: Number of validation steps to show
            
        Returns:
            List of generated plot file paths
        """
        generated_plots = []
        
        try:
            # Generate time vs pair heatmaps
            heatmap_path = self.plot_time_vs_pair_heatmaps(
                processed_data.ground_truth_full,
                processed_data.predictions_full,
                processed_data.valid_set_processed,
                num_pairs_to_show,
                valid_steps_to_show
            )
            generated_plots.append(heatmap_path)
            
            self.logger.info(f"Generated {len(generated_plots)} heatmap plots")
            
        except Exception as e:
            self.logger.error(f"Failed to generate heatmap plots: {e}")
        
        return generated_plots
    
    def plot_time_vs_pair_heatmaps(
        self,
        gt_full: pd.DataFrame,
        pred_full: pd.DataFrame,
        valid_set_post: pd.DataFrame,
        num_pairs_to_show: int = 50,
        valid_steps_to_show: int = 20
    ) -> str:
        """Plot vertically stacked heatmaps of GT, Predictions, and Overlay.
        
        Args:
            gt_full: Ground truth full interaction grid
            pred_full: Predictions full interaction grid
            valid_set_post: Validation set data
            num_pairs_to_show: Number of pairs to display
            valid_steps_to_show: Number of validation steps to show
            
        Returns:
            Path to saved plot
        """
        self.logger.info("Generating time vs pair heatmaps")
        
        # Apply publication style
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
            'axes.spines.bottom': True
        })
        
        # Okabe-Ito Colors for Overlay
        okabe_ito = {
            'vermilion': '#D55E00',
            'blue': '#0072B2',
            'bluish_green': '#009E73',
            'orange': '#E69F00',
            'sky_blue': '#56B4E9',
            'white': '#FFFFFF',
            'light_gray': '#F0F0F0',
            'dark_gray': '#2C3E50'
        }
        
        # Process test data
        gt_pivot_test = gt_full.pivot(index='pair', columns='time_stamp', values='present')
        pred_pivot_test = pred_full.pivot(index='pair', columns='time_stamp', values='present')
        
        common_pairs = gt_pivot_test.index.intersection(pred_pivot_test.index)
        gt_pivot_test = gt_pivot_test.loc[common_pairs].fillna(0).astype(int)
        pred_pivot_test = pred_pivot_test.loc[common_pairs].fillna(0).astype(int)
        
        # Select pairs to show
        if len(common_pairs) > num_pairs_to_show:
            pair_freq = gt_pivot_test.sum(axis=1).sort_values(ascending=False)
            selected_pairs = pair_freq.head(num_pairs_to_show).index
            gt_pivot_test = gt_pivot_test.loc[selected_pairs]
            pred_pivot_test = pred_pivot_test.loc[selected_pairs]
            plot_title_suffix = f' (Top {num_pairs_to_show} Pairs)'
        else:
            selected_pairs = common_pairs
            plot_title_suffix = ' (All Common Pairs)'
        
        # Sort pairs for better visualization
        try:
            sorted_index = gt_pivot_test.index.map(
                lambda x: tuple(self.custom_sort_residues(p) for p in x.split('_'))
            )
            gt_pivot_test = gt_pivot_test.loc[sorted_index.sort_values().index]
            pred_pivot_test = pred_pivot_test.loc[gt_pivot_test.index]
        except Exception as e:
            self.logger.warning(f"Custom pair sorting failed ({e}). Using simple string sort.")
            gt_pivot_test = gt_pivot_test.sort_index()
            pred_pivot_test = pred_pivot_test.loc[gt_pivot_test.index]
        
        # Create overlay matrix
        overlay_matrix_test = pd.DataFrame(0, index=gt_pivot_test.index, columns=gt_pivot_test.columns)
        overlay_matrix_test[(gt_pivot_test == 1) & (pred_pivot_test == 0)] = 1  # FN
        overlay_matrix_test[(gt_pivot_test == 0) & (pred_pivot_test == 1)] = 2  # FP
        overlay_matrix_test[(gt_pivot_test == 1) & (pred_pivot_test == 1)] = 3  # TP
        
        # Process validation data
        valid_pivot = pd.DataFrame()
        last_valid_timestamps = []
        
        if not valid_set_post.empty and valid_steps_to_show > 0:
            all_valid_timestamps = sorted(valid_set_post['time_stamp'].unique())
            if len(all_valid_timestamps) >= valid_steps_to_show:
                last_valid_timestamps = all_valid_timestamps[-valid_steps_to_show:]
                valid_data_filtered = valid_set_post[
                    (valid_set_post['time_stamp'].isin(last_valid_timestamps)) &
                    (valid_set_post['pair'].isin(selected_pairs))
                ]
                
                if not valid_data_filtered.empty:
                    valid_data_filtered = valid_data_filtered.assign(present=1)
                    valid_pivot = valid_data_filtered.pivot_table(
                        index='pair', columns='time_stamp', values='present', fill_value=0
                    )
                    valid_pivot = valid_pivot.reindex(gt_pivot_test.index, fill_value=0)
                    valid_pivot = valid_pivot.reindex(sorted(valid_pivot.columns), axis=1)
        
        # Combine data
        if not valid_pivot.empty:
            gt_combined = pd.concat([valid_pivot, gt_pivot_test], axis=1)
            pred_combined = pd.concat([valid_pivot, pred_pivot_test], axis=1)
            overlay_valid_part = valid_pivot.replace({0: 0, 1: 3})
            overlay_combined = pd.concat([overlay_valid_part, overlay_matrix_test], axis=1)
            valid_data_offset = len(last_valid_timestamps)
        else:
            gt_combined = gt_pivot_test
            pred_combined = pred_pivot_test
            overlay_combined = overlay_matrix_test
            valid_data_offset = 0
        
        # Define colors using Okabe-Ito palette
        # TN = White/LightGray, FN = Vermilion, FP = Orange, TP = BluishGreen
        
        # Ground Truth: 0=White, 1=Blue
        cmap_gt = mcolors.ListedColormap([okabe_ito['light_gray'], okabe_ito['blue']])
        
        # Predictions: 0=White, 1=SkyBlue
        cmap_pred = mcolors.ListedColormap([okabe_ito['light_gray'], okabe_ito['sky_blue']])
        
        # Overlay: 0=TN(LightGray), 1=FN(Vermilion), 2=FP(Orange), 3=TP(BluishGreen)
        cmap_overlay = mcolors.ListedColormap([
            okabe_ito['light_gray'],  # TN
            okabe_ito['vermilion'],   # FN
            okabe_ito['orange'],      # FP
            okabe_ito['bluish_green'] # TP
        ])
        
        overlay_labels = [
            'TN (Correct Negative)', 'FN (Missed)', 
            'FP (False Positive)', 'TP (Correct Positive)'
        ]
        
        # Create figure
        fig_height = max(12, len(gt_combined.index) * 0.6)
        fig_width = max(15, gt_combined.shape[1] * 0.25)
        fig, axes = plt.subplots(3, 1, figsize=(fig_width, fig_height), sharex=False, sharey=True)
        
        common_heatmap_kws = {"linewidths": 0.1, "linecolor": 'white'} # changed to white for cleaner look
        current_yticklabels = gt_combined.index
        
        # Plot Ground Truth
        sns.heatmap(
            gt_combined, ax=axes[0], cmap=cmap_gt, cbar=False,
            yticklabels=current_yticklabels, **common_heatmap_kws
        )
        axes[0].set_title('Ground Truth (Last Validation + Test)', fontweight='bold')
        axes[0].set_ylabel('Residue Pair', fontweight='bold')
        axes[0].set_xlabel('')
        
        # Plot Predictions
        sns.heatmap(
            pred_combined.fillna(0).astype(int), ax=axes[1], cmap=cmap_pred,
            vmin=0, vmax=1, cbar=False, yticklabels=current_yticklabels, **common_heatmap_kws
        )
        axes[1].set_title('Predictions (Validation GT + Test Predictions)', fontweight='bold')
        axes[1].set_ylabel('Residue Pair', fontweight='bold')
        axes[1].set_xlabel('')
        
        # Plot Overlay
        bounds_overlay = [0, 1, 2, 3, 4]
        norm_overlay = mcolors.BoundaryNorm(bounds_overlay, cmap_overlay.N)
        
        cax = sns.heatmap(
            overlay_combined.fillna(0).astype(int), ax=axes[2], cmap=cmap_overlay,
            norm=norm_overlay, cbar=True, yticklabels=current_yticklabels,
            **common_heatmap_kws, cbar_kws={"ticks": [0.5, 1.5, 2.5, 3.5], "label": "Result Type"}
        )
        axes[2].set_title('Overlay (Validation TN/TP + Test Result)', fontweight='bold')
        axes[2].set_xlabel('Time Stamp', fontweight='bold')
        axes[2].set_ylabel('Residue Pair', fontweight='bold')
        
        # Set colorbar labels
        colorbar = cax.collections[0].colorbar
        colorbar.set_ticklabels(overlay_labels)
        colorbar.ax.tick_params(labelsize=18)
        
        # Adjust font sizes
        num_labels = len(current_yticklabels)
        font_size = max(8, min(16, int((fig_height / num_labels) * 72 * 0.35))) if num_labels > 0 else 10
        
        for ax in axes:
            ax.tick_params(axis='y', labelsize=font_size)
            ax.tick_params(axis='x', labelsize=16)
        
        # Add vertical separator if validation data included
        if valid_data_offset > 0:
            for ax in axes:
                ax.axvline(x=valid_data_offset, color=okabe_ito['dark_gray'], linestyle='--', linewidth=2)
                ax.text(
                    valid_data_offset / 2., ax.get_ylim()[0] * 1.02, 'Validation',
                    ha='center', va='bottom', color=okabe_ito['vermilion'], fontsize=18, weight='bold'
                )
                ax.text(
                    valid_data_offset + (gt_combined.shape[1] - valid_data_offset) / 2.,
                    ax.get_ylim()[0] * 1.02, 'Test',
                    ha='center', va='bottom', color=okabe_ito['dark_gray'], fontsize=18, weight='bold'
                )
        
        fig.suptitle(
            f'Interaction Dynamics: Validation History vs Test Prediction{plot_title_suffix}',
            fontsize=32, fontweight='bold', y=0.995
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Save and close
        filename = 'heatmap_time_vs_pairs_VERTICAL_with_valid.png'
        plot_path = self.save_plot(filename, fig)
        
        # Also save SVG
        svg_filename = filename.replace('.png', '.svg')
        self.save_plot(svg_filename, fig)
        
        self.close_plot(fig)
        
        return plot_path

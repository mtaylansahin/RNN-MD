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
        valid_steps_to_show: int = 20  # Kept for compatibility but might be ignored if showing full history
    ) -> List[str]:
        """Generate all heatmap plots.
        
        Args:
            processed_data: Processed analysis data
            num_pairs_to_show: Number of pairs to show in heatmaps
            valid_steps_to_show: Number of validation steps to show (deprecated if showing full history)
            
        Returns:
            List of generated plot file paths
        """
        generated_plots = []
        
        try:
            # Generate time vs pair heatmaps
            heatmap_path = self.plot_time_vs_pair_heatmaps(
                processed_data.ground_truth_full,
                processed_data.predictions_full,
                processed_data.train_set_processed,
                processed_data.valid_set_processed,
                num_pairs_to_show
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
        train_set: pd.DataFrame,
        valid_set: pd.DataFrame,
        num_pairs_to_show: int = 50
    ) -> str:
        """Plot vertically stacked heatmaps of GT, Predictions, and Overlay.
        
        Args:
            gt_full: Ground truth full interaction grid
            pred_full: Predictions full interaction grid
            train_set: Training set data
            valid_set: Validation set data
            num_pairs_to_show: Number of pairs to display
            
        Returns:
            Path to saved plot
        """
        self.logger.info("Generating time vs pair heatmaps with full history")
        
        # Apply publication style
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'],
            'font.size': 14, # Reduced base font size
            'axes.titlesize': 20, # Reduced title size
            'axes.labelsize': 16, # Reduced label size
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 14,
            'figure.titlesize': 24, # Reduced figure title size
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
            'dark_gray': '#2C3E50',
            'neutral_gray': '#999999' # For history
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
        
        # Create overlay matrix for test
        overlay_matrix_test = pd.DataFrame(0, index=gt_pivot_test.index, columns=gt_pivot_test.columns)
        overlay_matrix_test[(gt_pivot_test == 1) & (pred_pivot_test == 0)] = 1  # FN
        overlay_matrix_test[(gt_pivot_test == 0) & (pred_pivot_test == 1)] = 2  # FP
        overlay_matrix_test[(gt_pivot_test == 1) & (pred_pivot_test == 1)] = 3  # TP
        
        # Process History (Train + Valid)
        history_pivot = pd.DataFrame()
        
        # Combine train and valid
        history_dfs = []
        if not train_set.empty:
            history_dfs.append(train_set)
        if not valid_set.empty:
            history_dfs.append(valid_set)
            
        if history_dfs:
            combined_history = pd.concat(history_dfs)
            # Filter for selected pairs
            combined_history = combined_history[combined_history['pair'].isin(selected_pairs)].copy()
            
            if not combined_history.empty:
                combined_history['present'] = 1
                history_pivot = combined_history.pivot_table(
                    index='pair', columns='time_stamp', values='present', fill_value=0
                )
                # Align index with test data
                history_pivot = history_pivot.reindex(gt_pivot_test.index, fill_value=0)
                # Sort columns (time)
                history_pivot = history_pivot.reindex(sorted(history_pivot.columns), axis=1)
        
        # Combine data
        if not history_pivot.empty:
            # History + Test GT
            gt_combined = pd.concat([history_pivot, gt_pivot_test], axis=1)
            # History + Test Preds (History is effectively GT here too as we don't plot predictions for it)
            pred_combined = pd.concat([history_pivot, pred_pivot_test], axis=1)
            
            # For overlay: History is shown as neutral (let's map it to 0 for now and use a different cmap, 
            # or map it to a specific value like 4 and handle it)
            # Actually easiest is to just treat it as "History" type
            # But the user wants history on the left for all plots.
            
            # Let's map history 1s to a specific value for the overlay plot if we want to color them differently
            # For the overlay plot, we want history to be "Neutral".
            # Let's use value 4 for "History Interaction"
            history_overlay = history_pivot.replace({0: 0, 1: 4}) # 4 = History Present
            
            overlay_combined = pd.concat([history_overlay, overlay_matrix_test], axis=1)
            history_offset = history_pivot.shape[1]
        else:
            gt_combined = gt_pivot_test
            pred_combined = pred_pivot_test
            overlay_combined = overlay_matrix_test
            history_offset = 0
            
        # Define Colors
        
        # History Color: Neutral Gray
        neutral_gray = okabe_ito['neutral_gray']
        
        # GT/Pred maps need to handle the "History" part if we want it to look the same
        # Actually, for GT and Pred plots, we can just use the standard colors, 
        # but maybe the user wants history to look "neutral" in ALL plots?
        # "Left side to show the full length... in a neutral color" implies all plots.
        
        # So we need custom color mapping for History part in all plots.
        # This is tricky with simple sns.heatmap unless we change values.
        
        # Strategy: Create a masking array or use RGB array. 
        # Or simpler: Change values in gt_combined/pred_combined for history columns to 2, 
        # and update colormap to have 3 colors: [Background, Test-Active, History-Active]
        
        if history_offset > 0:
            # Update history parts to value 0.5 (between 0 and 1) or 2
            # Let's use 2 for History Active
            gt_combined.iloc[:, :history_offset] = gt_combined.iloc[:, :history_offset].replace({1: 2})
            pred_combined.iloc[:, :history_offset] = pred_combined.iloc[:, :history_offset].replace({1: 2})
            
        # Ground Truth Map: 0=Bg, 1=Blue(Test), 2=Neutral(History)
        cmap_gt = mcolors.ListedColormap([
            okabe_ito['light_gray'], # 0
            okabe_ito['blue'],       # 1
            neutral_gray             # 2
        ])
        
        # Prediction Map: 0=Bg, 1=SkyBlue(Test), 2=Neutral(History)
        cmap_pred = mcolors.ListedColormap([
            okabe_ito['light_gray'], # 0
            okabe_ito['sky_blue'],   # 1
            neutral_gray             # 2
        ])
        
        # Overlay Map: 0=TN, 1=FN, 2=FP, 3=TP, 4=History(Neutral)
        cmap_overlay = mcolors.ListedColormap([
            okabe_ito['light_gray'],  # 0: TN
            okabe_ito['orange'],      # 1: FN 
            okabe_ito['vermilion'],   # 2: FP
            okabe_ito['bluish_green'],# 3: TP
            neutral_gray              # 4: History
        ])
        
        overlay_labels = [
            'TN', 'FN', 'FP', 'TP', 'History'
        ]
        
        # Create figure
        # Adjusted height for wider cells (more square-like)
        # Assuming ~300 time points and ~50 pairs, we need aspect ratio ~6:1
        # Increase width relative to height
        num_cols = gt_combined.shape[1]
        num_rows = len(gt_combined.index)
        
        # Calculate aspect ratio to make cells approximately square
        # We want width/height approx proportional to num_cols/num_rows
        # Base scale factor
        scale = 0.4
        # To make cells square: fig_width / num_cols ≈ fig_height / num_rows
        # fig_height = num_rows * scale
        # fig_width = num_cols * scale
        
        fig_height = max(12, num_rows * scale)
        fig_width = max(20, num_cols * scale) # Use same scale for width to get square-ish cells
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Increased font sizes - boosted by another 50%
        plt.rcParams.update({
            'font.size': 40,
            'axes.titlesize': 54,
            'axes.labelsize': 50,
            'xtick.labelsize': 40,
            'ytick.labelsize': 27, # Kept same as requested (but calculation below controls Y labels)
            'legend.fontsize': 40,
        })
        
        common_heatmap_kws = {"linewidths": 0.05, "linecolor": 'white', "square": False} # Set square=False to allow manual aspect adjustment if needed, but fig size helps
        current_yticklabels = gt_combined.index
        
        # Plot Overlay
        bounds_overlay = [0, 1, 2, 3, 4, 5]
        norm_overlay = mcolors.BoundaryNorm(bounds_overlay, cmap_overlay.N)
        
        cax = sns.heatmap(
            overlay_combined.fillna(0).astype(int), ax=ax, cmap=cmap_overlay,
            norm=norm_overlay, cbar=True, yticklabels=current_yticklabels,
            **common_heatmap_kws, cbar_kws={"ticks": [0.5, 1.5, 2.5, 3.5, 4.5], "label": "Result Type", "pad": 0.02}
        )
        
        # Remove title
        ax.set_title('') 
        
        # Labels moved down with labelpad, removed bold weight
        ax.set_xlabel('Simulation time (ns)', fontsize=50, labelpad=60)
        ax.set_ylabel('Residue Pair', fontsize=50)
        
        # Set colorbar labels
        colorbar = cax.collections[0].colorbar
        colorbar.set_ticklabels(overlay_labels)
        colorbar.ax.tick_params(labelsize=45)
        colorbar.set_label("Result Type", fontsize=50)
        
        # Adjust font sizes for Y axis (Residue Pairs)
        num_labels = len(current_yticklabels)
        # Kept same as requested
        font_size = max(15, min(27, int((fig_height / num_labels) * 72 * 0.45))) if num_labels > 0 else 21
        
        ax.tick_params(axis='y', labelsize=font_size)
        
        # Handle X-axis Ticks (Convert to ns)
        # 1 timestep = 0.5 ns
        # We want ticks every 5 ns
        # 5 ns = 10 timesteps (since 10 * 0.5 = 5.0)
        timestamps = gt_combined.columns
        
        # Create formatter for x-axis
        def time_formatter(x, pos):
            try:
                idx = int(x)
                if 0 <= idx < len(timestamps):
                    ts = timestamps[idx]
                    return f"{ts * 0.5:.0f}" # Convert to ns, no decimals for cleaner look
                return ""
            except:
                return ""
                
        # Set ticks explicitly every 10 indices (which is 5ns)
        from matplotlib.ticker import FuncFormatter, MultipleLocator
        ax.xaxis.set_major_formatter(FuncFormatter(time_formatter))
        # Use MultipleLocator(10) to place a tick every 10 data points (timesteps)
        # This corresponds to exactly 5ns intervals
        ax.xaxis.set_major_locator(MultipleLocator(10)) 
        ax.tick_params(axis='x', labelsize=45, rotation=0)

        # Add vertical separator for History vs Test
        if history_offset > 0:
            ax.axvline(x=history_offset, color=okabe_ito['dark_gray'], linestyle='--', linewidth=3)
            
            # Add text labels - moved down (approx 1.05 * ylim[0])
            # Note: ylim[0] is usually the bottom (max index) for heatmaps. 
            # Check orientation: if origin is upper, ylim is (bottom, top) = (max_y, 0).
            # So ylim[0] is the bottom edge.
            
            label_y_pos = ax.get_ylim()[0] * 1.05
            
            # History Label
            ax.text(
                history_offset / 2., label_y_pos, 'History (Train+Val)',
                ha='center', va='top', color=okabe_ito['dark_gray'], fontsize=45
            )
            
            # Test Label
            ax.text(
                history_offset + (gt_combined.shape[1] - history_offset) / 2.,
                label_y_pos, 'Test',
                ha='center', va='top', color=okabe_ito['dark_gray'], fontsize=45
            )
        
        # Remove suptitle as well
        # fig.suptitle(...) 
        
        plt.tight_layout()
        
        # Save and close
        filename = 'heatmap_time_vs_pairs_VERTICAL_full_history.png'
        plot_path = self.save_plot(filename, fig)
        
        # Also save SVG
        svg_filename = filename.replace('.png', '.svg')
        self.save_plot(svg_filename, fig)
        
        self.close_plot(fig)
        
        return plot_path

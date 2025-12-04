#!/usr/bin/env python3
"""
Temporal Dynamics and Interaction Stability Analysis

This script provides focused analysis on:
1. Temporal dynamics with separate plots for each complex system
2. Interaction stability distribution analysis with 2x2 grid layout

Combines the best visualizations from comprehensive_protein_analysis.py and improved_analysis_plots.py
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from collections import defaultdict, Counter
import warnings
from matplotlib.patches import Patch, Rectangle
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

# Publication-ready styling configuration
plt.rcParams.update({
    # Font settings for publication quality
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    
    # Professional plot styling
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.linewidth': 1.2,
    'axes.edgecolor': '#2C3E50',
    
    # Grid and background
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.6,
    'grid.color': '#BDC3C7',
    'axes.axisbelow': True,
    
    # Lines and patches
    'lines.linewidth': 2.5,
    'lines.solid_capstyle': 'round',
    'patch.linewidth': 0.8,
    'patch.edgecolor': 'white',
    
    # Figure settings
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.edgecolor': 'none',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    
    # Text and annotations
    'text.color': '#2C3E50',
    'axes.labelcolor': '#2C3E50',
    'xtick.color': '#2C3E50',
    'ytick.color': '#2C3E50',
    
    # Legend styling
    'legend.frameon': True,
    'legend.fancybox': True,
    'legend.shadow': False,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '#BDC3C7',
    'legend.borderpad': 0.5
})

class TemporalStabilityAnalysis:
    """
    Focused analysis class for temporal dynamics and interaction stability
    """
    
    def __init__(self, data_dir="data", output_dir="temporal_stability_results"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Data storage
        self.all_data = {}
        
        # Publication-ready color palettes
        self.stability_palette = {
            'Rare': '#E74C3C',      # Vivid red for rare interactions
            'Transient': '#F39C12',  # Warm amber for transient interactions
            'Stable': '#27AE60'     # Rich green for stable interactions
        }
        
        # Additional publication colors
        self.publication_colors = {
            'primary': '#2C3E50',    # Dark blue-gray for text/lines
            'secondary': '#34495E',  # Lighter blue-gray for accents
            'background': '#ECF0F1', # Light gray for backgrounds
            'grid': '#BDC3C7',       # Medium gray for grids
            'highlight': '#3498DB'   # Blue for highlights
        }
        
        # Colors for different replicas
        self.replica_colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6', '#E67E22', '#1ABC9C', '#34495E']
        
        print(f"Initialized Temporal & Stability Analysis for data in: {self.data_dir}")
        print(f"Results will be saved to: {self.output_dir}")
    
    def load_all_data(self):
        """Load all interaction data from all complexes and replicas"""
        print("Loading all interaction data...")
        
        for complex_dir in self.data_dir.iterdir():
            if complex_dir.is_dir() and not complex_dir.name.startswith('.'):
                complex_name = complex_dir.name
                print(f"Processing complex: {complex_name}")
                
                self.all_data[complex_name] = {}
                
                for replica_dir in complex_dir.iterdir():
                    if replica_dir.is_dir() and replica_dir.name.startswith('replica'):
                        replica_name = replica_dir.name
                        print(f"  Processing {replica_name}")
                        
                        # Find the interfacea directory
                        interfacea_dirs = list(replica_dir.glob("*interfacea*"))
                        if interfacea_dirs:
                            interfacea_dir = interfacea_dirs[0]
                            data = self._load_replica_data(interfacea_dir)
                            self.all_data[complex_name][replica_name] = data
                        
        print(f"Loaded data for {len(self.all_data)} complexes")
        return self.all_data
    
    def _load_replica_data(self, interfacea_dir):
        """Load data from a single replica"""
        replica_data = []
        
        interfacea_files = list(interfacea_dir.glob("*.interfacea"))
        interfacea_files.sort(key=lambda x: int(re.findall(r'\d+', x.name)[0]))
        
        for i, file_path in enumerate(interfacea_files):
            try:
                # Extract time stamp from filename
                time_stamp = int(re.findall(r'\d+', file_path.name)[0])
                
                # Read the file
                df = pd.read_csv(file_path, sep=r'\s+', header=0,
                               names=['itype', 'chain_a', 'chain_b', 'resname_a', 'resname_b',
                                     'resid_a', 'resid_b', 'atom_a', 'atom_b'])
                
                if not df.empty:
                    df['time_stamp'] = time_stamp
                    df['snapshot'] = i
                    replica_data.append(df)
                    
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        if replica_data:
            return pd.concat(replica_data, ignore_index=True)
        else:
            return pd.DataFrame()

    def analyze_temporal_dynamics(self):
        """
        Analyze temporal dynamics with smoothed visualizations - separate plots for each complex
        Modified to create individual plots for each protein complex system
        """
        print("=" * 80)
        print("TEMPORAL DYNAMICS ANALYSIS")
        print("=" * 80)
        
        window_size = 10  # Rolling window for smoothing
        
        # Analyze each complex separately
        for complex_idx, (complex_name, complex_data) in enumerate(self.all_data.items()):
            if not complex_data:
                continue
                
            print(f"Creating temporal dynamics plot for {complex_name}...")
            
            # Create separate figure for each complex - simplified 1x2 layout
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            complex_display_name = complex_name.replace('_interchain', '')
            fig.suptitle(f'Temporal Dynamics - {complex_display_name}', fontsize=16, fontweight='bold')
            
            # Collect data for this complex
            complex_interface_stability = []
            
            # Process each replica for this complex
            for replica_idx, (replica_name, replica_data) in enumerate(complex_data.items()):
                if replica_data.empty:
                    continue
                
                color = self.replica_colors[replica_idx % len(self.replica_colors)]
                
                # 1. Total interactions over time (smoothed)
                #    Deduplicate duplicate interactions within the same timestep
                #    (same interaction type between the same residues; A↔B treated as the same pair)
                dedup_df = replica_data.copy()
                # Normalize residue pair ordering so (A,B) and (B,A) map to the same key
                chains_a = dedup_df['chain_a'].astype(str)
                chains_b = dedup_df['chain_b'].astype(str)
                resid_a_int = dedup_df['resid_a'].astype(int)
                resid_b_int = dedup_df['resid_b'].astype(int)
                a_first = (chains_a < chains_b) | ((chains_a == chains_b) & (resid_a_int <= resid_b_int))
                dedup_df['n_chain1'] = np.where(a_first, chains_a, chains_b)
                dedup_df['n_resid1'] = np.where(a_first, resid_a_int, resid_b_int)
                dedup_df['n_chain2'] = np.where(a_first, chains_b, chains_a)
                dedup_df['n_resid2'] = np.where(a_first, resid_b_int, resid_a_int)

                unique_per_ts = dedup_df.drop_duplicates(
                    subset=['time_stamp', 'itype', 'n_chain1', 'n_resid1', 'n_chain2', 'n_resid2']
                )
                time_counts = unique_per_ts.groupby('time_stamp').size()
                if len(time_counts) > window_size:
                    time_counts_smooth = time_counts.rolling(window=window_size, center=True).mean()
                else:
                    time_counts_smooth = time_counts
                
                # Plot raw data as background
                axes[0].plot(time_counts.index, time_counts.values, 
                              alpha=0.2, linewidth=1, color=color)
                # Plot smoothed data
                axes[0].plot(time_counts.index, time_counts_smooth.values, 
                              label=f"{replica_name}", linewidth=2, alpha=0.8, color=color)
                
                # 2. Interface stability over time
                #    Count unique interface residues per timestep (deduplicated A↔B pairs do not matter here)
                interface_residues = unique_per_ts.groupby('time_stamp').apply(
                    lambda x: len(set(x['n_resid1']).union(set(x['n_resid2']))))
                
                if len(interface_residues) > window_size:
                    interface_smooth = interface_residues.rolling(window=window_size, center=True).mean()
                else:
                    interface_smooth = interface_residues
                
                axes[1].plot(interface_residues.index, interface_smooth.values, 
                              label=f"{replica_name}", linewidth=2, alpha=0.8, color=color)
                
                # Store for complex-specific analysis
                complex_interface_stability.extend(interface_residues.values)
            
            # Style the temporal plots for this complex - simplified 1x2 layout
            axes[0].set_title(f'Total Interactions Over Time\n(Smoothed, window={window_size})', fontweight='bold')
            axes[0].set_xlabel('Time Stamp')
            axes[0].set_ylabel('Number of Interactions')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            axes[1].set_title('Interface Size Stability\n(Number of Interface Residues)', fontweight='bold')
            axes[1].set_xlabel('Time Stamp')
            axes[1].set_ylabel('Interface Size')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # Save separate file for each complex
            plt.tight_layout()
            plt.savefig(self.output_dir / f'temporal_dynamics_{complex_name}.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        print(f"✓ Generated separate temporal dynamics plots for each complex system")

    def plot_interaction_stability(self):
        """
        Generate enhanced interaction stability analysis with grouped layout for publication.
        Creates a single figure with 2x4 grid (2 complexes × 4 replicas) for efficient space usage.
        """
        print("\n🎯 Generating publication-ready interaction stability plot with grouped layout...")
        
        # Get list of complexes and ensure consistent ordering
        complex_names = sorted(list(self.all_data.keys()))
        # Put 1JPS on top (first row) if present
        complex_names = sorted(complex_names, key=lambda n: 0 if '1JPS' in n.upper() else 1)
        if len(complex_names) == 0:
            print("No complexes found for stability analysis")
            return
            
        print(f"Creating grouped stability plot for {len(complex_names)} complexes...")
        
        # Create publication-ready figure with 2x4 layout - increased height for better spacing
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(2, 4, hspace=0.45, wspace=0.25, 
                     left=0.06, right=0.94, top=0.85, bottom=0.15)
        
        # Enhanced publication title with better positioning
        fig.suptitle(
            "Protein-Protein Interface Interaction Stability Analysis",
            fontsize=18, fontweight='bold', y=0.92
        )
        
        # Global statistics for summary
        all_complex_stats = {}
        
        # Plot each complex as a row
        for complex_idx, complex_name in enumerate(complex_names):
            replicas = self.all_data[complex_name]
            if not replicas:
                continue
            
            complex_display_name = complex_name.replace('_interchain', '')
            replica_list = sorted(list(replicas.items()))
            
            # Store stats for this complex
            complex_stats = {
                'rare_counts': [],
                'transient_counts': [],
                'stable_counts': [],
                'total_pairs': [],
                'mean_frequencies': [],
                'median_frequencies': []
            }
            
            # Track axes and max y for standardizing y-axis within this complex
            row_axes = []
            row_max_count = 0.0
            
            # Plot each replica as a column
            for replica_idx, (replica_name, data) in enumerate(replica_list):
                if replica_idx >= 4:  # Limit to 4 replicas
                    break
                    
                ax = fig.add_subplot(gs[complex_idx, replica_idx])
                row_axes.append(ax)
                
                if data.empty:
                    ax.text(0.5, 0.5, "No Data\nAvailable", 
                           ha="center", va="center", fontsize=12, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                    ax.set_title(f"{complex_display_name}\n{replica_name}", 
                               fontsize=12, fontweight='bold')
                    continue

                # Calculate interaction frequencies for this replica
                total_timepoints = data["time_stamp"].nunique()
                
                # Remove exact duplicates from raw data before processing
                data_clean = data.drop_duplicates()
                
                # Count UNIQUE timepoints where each pair appears (not total occurrences)
                pair_timepoint_counts = data_clean.groupby(["resid_a", "resid_b"])["time_stamp"].nunique()
                frequencies = (pair_timepoint_counts / total_timepoints) * 100

                # Publication-quality histogram with 5% bin width (0,5,...,100)
                bins = np.arange(0, 105, 5)
                counts, bin_edges, patches = ax.hist(
                    frequencies, bins=bins, alpha=0.8, 
                    edgecolor='white', linewidth=0.6, density=False
                )

                # Apply consistent color coding
                for patch, bin_start, bin_end in zip(patches, bin_edges[:-1], bin_edges[1:]):
                    bin_center = (bin_start + bin_end) / 2
                    if bin_center < 5:
                        patch.set_facecolor(self.stability_palette['Rare'])
                    elif bin_center < 50:
                        patch.set_facecolor(self.stability_palette['Transient'])
                    else:
                        patch.set_facecolor(self.stability_palette['Stable'])
                
                # Track max count for y-axis standardization within this complex
                if len(counts) > 0:
                    row_max_count = max(row_max_count, float(counts.max()))

                # Add reference lines with publication styling
                ax.axvline(5, color="#34495E", linestyle="--", alpha=0.7, linewidth=1.5)
                ax.axvline(50, color="#34495E", linestyle="--", alpha=0.7, linewidth=1.5)
                
                # Calculate statistics
                rare_count = (frequencies < 5).sum()
                transient_count = ((frequencies >= 5) & (frequencies <= 50)).sum()
                stable_count = (frequencies > 50).sum()
                total_pairs = len(frequencies)

                # Store stats for complex summary
                complex_stats['rare_counts'].append(rare_count)
                complex_stats['transient_counts'].append(transient_count)
                complex_stats['stable_counts'].append(stable_count)
                complex_stats['total_pairs'].append(total_pairs)
                complex_stats['mean_frequencies'].append(frequencies.mean())
                complex_stats['median_frequencies'].append(frequencies.median())

                # Enhanced titles with complex and replica information
                ax.set_title(f"{complex_display_name}\n{replica_name}", 
                           fontsize=11, fontweight='bold', pad=8)
                
                # Only add x-label to bottom row
                if complex_idx == len(complex_names) - 1:
                    ax.set_xlabel("Interaction Frequency (%)", fontsize=11)
                    
                # Only add y-label to leftmost column
                if replica_idx == 0:
                    ax.set_ylabel("Count", fontsize=11)
                
                # Compact statistical annotation
                stats_text = f"n={total_pairs:,}\nμ={frequencies.mean():.1f}%"
                props = dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.85, edgecolor='gray', linewidth=0.5)
                ax.text(0.97, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                       verticalalignment='top', horizontalalignment='right', bbox=props)
                
                # Minimal grid for publication
                ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
                ax.set_xlim(0, 100)
                
                # y-axis limits will be standardized across replicas for this complex below
            
            # Standardize y-axis across replicas for this complex (row)
            uniform_ylim_max = (row_max_count * 1.1) if row_max_count > 0 else 1
            for row_ax in row_axes:
                row_ax.set_ylim(0, uniform_ylim_max)
                # Ensure y-axis uses integer ticks for counts
                row_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            
            # Store complex stats for summary
            all_complex_stats[complex_name] = complex_stats
        
        # Create publication-quality legend
        legend_elements = [
            Patch(facecolor=self.stability_palette['Rare'], 
                  label="Rare Interactions (<5%)", alpha=0.8),
            Patch(facecolor=self.stability_palette['Transient'], 
                  label="Transient Interactions (5-50%)", alpha=0.8),
            Patch(facecolor=self.stability_palette['Stable'], 
                  label="Stable Interactions (>50%)", alpha=0.8),
        ]
        
        # Position legend below the plots with proper spacing
        fig.legend(handles=legend_elements, 
                  loc='lower center', 
                  bbox_to_anchor=(0.5, 0.05),
                  ncol=3, 
                  frameon=True, 
                  fancybox=True, 
                  shadow=True,
                  fontsize=12)
        
        # Add axis labels with better positioning
        fig.text(0.02, 0.5, 'Interaction Count', rotation=90, ha='center', va='center', fontsize=14, fontweight='bold')
        fig.text(0.5, 0.10, 'Interaction Frequency (%)', ha='center', va='center', fontsize=14, fontweight='bold')
        
        # Apply tight layout with proper margins for the new spacing
        plt.tight_layout(rect=[0.04, 0.12, 0.96, 0.90])
        
        # Save high-quality publication figure
        plt.savefig(self.output_dir / 'interaction_stability_2x4_grid.png', 
                   dpi=300, bbox_inches='tight', facecolor='white', 
                   format='png')
        
        # Also save as vector format for publications
        plt.savefig(self.output_dir / 'interaction_stability_2x4_grid.pdf', 
                   bbox_inches='tight', facecolor='white', format='pdf')
        
        plt.show()
        
        # Print summary statistics
        print(f"\n📊 PUBLICATION SUMMARY:")
        print("=" * 60)
        for complex_name, stats in all_complex_stats.items():
            if stats['total_pairs']:
                display_name = complex_name.replace('_interchain', '')
                total_pairs = sum(stats['total_pairs'])
                avg_rare = np.mean(stats['rare_counts'])
                avg_transient = np.mean(stats['transient_counts'])
                avg_stable = np.mean(stats['stable_counts'])
                
                print(f"{display_name}:")
                print(f"  Total interaction pairs: {total_pairs:,}")
                print(f"  Average per replica - Rare: {avg_rare:.1f}, Transient: {avg_transient:.1f}, Stable: {avg_stable:.1f}")
                print(f"  Mean frequency: {np.mean(stats['mean_frequencies']):.1f}% ± {np.std(stats['mean_frequencies']):.1f}%")
        
        print(f"\n✅ Generated publication-ready grouped stability plot (2×4 grid)")
        print(f"📁 Saved as PNG (raster) and PDF (vector) formats")
        
        return all_complex_stats  # Return stats for potential cross-complex analysis

    def plot_cross_complex_comparison(self, complex_stats=None):
        """
        Generate a cross-complex comparison plot for publication showing statistical comparisons.
        """
        if complex_stats is None:
            print("No statistics provided. Run plot_interaction_stability first.")
            return
            
        print("\n📊 Generating cross-complex comparison plot...")
        
        # Create publication-quality comparison figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Cross-Complex Interaction Stability Comparison", 
                    fontsize=18, fontweight='bold', y=0.95)
        
        # Prepare data for comparison
        complex_names = list(complex_stats.keys())
        display_names = [name.replace('_interchain', '') for name in complex_names]
        
        # 1. Stability category proportions
        ax1 = axes[0, 0]
        rare_props = []
        transient_props = []
        stable_props = []
        
        for complex_name in complex_names:
            stats = complex_stats[complex_name]
            if stats['total_pairs']:
                total_pairs = sum(stats['total_pairs'])
                rare_prop = (sum(stats['rare_counts']) / total_pairs) * 100
                transient_prop = (sum(stats['transient_counts']) / total_pairs) * 100
                stable_prop = (sum(stats['stable_counts']) / total_pairs) * 100
                
                rare_props.append(rare_prop)
                transient_props.append(transient_prop)
                stable_props.append(stable_prop)
        
        # Stacked bar plot
        width = 0.6
        x_pos = np.arange(len(display_names))
        
        bars1 = ax1.bar(x_pos, rare_props, width, 
                       label='Rare (<10%)', color=self.stability_palette['Rare'], alpha=0.8)
        bars2 = ax1.bar(x_pos, transient_props, width, bottom=rare_props,
                       label='Transient (10-50%)', color=self.stability_palette['Transient'], alpha=0.8)
        bars3 = ax1.bar(x_pos, stable_props, width, 
                       bottom=np.array(rare_props) + np.array(transient_props),
                       label='Stable (>50%)', color=self.stability_palette['Stable'], alpha=0.8)
        
        ax1.set_xlabel('Protein Complex', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Proportion of Interactions (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Interaction Stability Distribution', fontsize=14, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(display_names)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels on bars
        for i, (rare, trans, stable) in enumerate(zip(rare_props, transient_props, stable_props)):
            if rare > 5:  # Only show if segment is large enough
                ax1.text(i, rare/2, f'{rare:.1f}%', ha='center', va='center', 
                        fontsize=10, fontweight='bold', color='white')
            if trans > 5:
                ax1.text(i, rare + trans/2, f'{trans:.1f}%', ha='center', va='center', 
                        fontsize=10, fontweight='bold', color='white')
            if stable > 5:
                ax1.text(i, rare + trans + stable/2, f'{stable:.1f}%', ha='center', va='center', 
                        fontsize=10, fontweight='bold', color='white')
        
        # 2. Mean frequency comparison with error bars
        ax2 = axes[0, 1]
        mean_freqs = []
        std_freqs = []
        
        for complex_name in complex_names:
            stats = complex_stats[complex_name]
            if stats['mean_frequencies']:
                mean_freqs.append(np.mean(stats['mean_frequencies']))
                std_freqs.append(np.std(stats['mean_frequencies']))
        
        bars = ax2.bar(x_pos, mean_freqs, width, yerr=std_freqs, capsize=5,
                      color=self.publication_colors['highlight'], alpha=0.7,
                      error_kw={'linewidth': 2, 'capthick': 2})
        
        ax2.set_xlabel('Protein Complex', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Mean Interaction Frequency (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Average Interaction Frequency', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(display_names)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(mean_freqs, std_freqs)):
            ax2.text(i, mean + std + 1, f'{mean:.1f}±{std:.1f}%', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 3. Total interaction pairs comparison
        ax3 = axes[1, 0]
        total_pairs_per_complex = []
        
        for complex_name in complex_names:
            stats = complex_stats[complex_name]
            if stats['total_pairs']:
                total_pairs_per_complex.append(sum(stats['total_pairs']))
        
        bars = ax3.bar(x_pos, total_pairs_per_complex, width,
                      color=self.publication_colors['secondary'], alpha=0.7)
        
        ax3.set_xlabel('Protein Complex', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Total Interaction Pairs', fontsize=12, fontweight='bold')
        ax3.set_title('Data Volume Comparison', fontsize=14, fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(display_names)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, total in enumerate(total_pairs_per_complex):
            ax3.text(i, total + max(total_pairs_per_complex) * 0.01, f'{total:,}', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 4. Statistical summary text
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create summary statistics text
        summary_text = []
        summary_text.append("CROSS-COMPLEX STATISTICAL SUMMARY")
        summary_text.append("=" * 40)
        summary_text.append("")
        
        for i, complex_name in enumerate(complex_names):
            stats = complex_stats[complex_name]
            display_name = display_names[i]
            
            if stats['total_pairs']:
                total_pairs = sum(stats['total_pairs'])
                mean_freq = np.mean(stats['mean_frequencies'])
                std_freq = np.std(stats['mean_frequencies'])
                
                # Stability proportions
                rare_prop = (sum(stats['rare_counts']) / total_pairs) * 100
                transient_prop = (sum(stats['transient_counts']) / total_pairs) * 100  
                stable_prop = (sum(stats['stable_counts']) / total_pairs) * 100
                
                summary_text.append(f"{display_name}:")
                summary_text.append(f"  Total pairs: {total_pairs:,}")
                summary_text.append(f"  Mean frequency: {mean_freq:.1f}% ± {std_freq:.1f}%")
                summary_text.append(f"  Stability: {rare_prop:.1f}% rare, {transient_prop:.1f}% transient, {stable_prop:.1f}% stable")
                summary_text.append("")
        
        # Statistical comparison
        if len(mean_freqs) == 2:
            from scipy import stats
            try:
                # Perform statistical test if scipy is available
                t_stat, p_value = stats.ttest_ind(
                    complex_stats[complex_names[0]]['mean_frequencies'],
                    complex_stats[complex_names[1]]['mean_frequencies']
                )
                summary_text.append("STATISTICAL COMPARISON:")
                summary_text.append(f"t-statistic: {t_stat:.3f}")
                summary_text.append(f"p-value: {p_value:.3f}")
                summary_text.append(f"Significance: {'**' if p_value < 0.01 else '*' if p_value < 0.05 else 'n.s.'}")
            except ImportError:
                summary_text.append("STATISTICAL COMPARISON:")
                summary_text.append("(scipy not available for statistical tests)")
        
        # Display summary text
        ax4.text(0.05, 0.95, '\n'.join(summary_text), 
                transform=ax4.transAxes, fontsize=10, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor=self.publication_colors['background'], alpha=0.8))
        
        # Save publication-quality comparison plot
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.savefig(self.output_dir / 'cross_complex_comparison.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(self.output_dir / 'cross_complex_comparison.pdf', 
                   bbox_inches='tight', facecolor='white', format='pdf')
        plt.show()
        
        print("✅ Generated cross-complex comparison plot for publication")

    def generate_summary_report(self):
        """Generate a summary report of the analysis"""
        report = []
        report.append("=" * 80)
        report.append("TEMPORAL DYNAMICS AND STABILITY ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        report.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Data overview
        report.append("1. DATA OVERVIEW")
        report.append("-" * 40)
        total_complexes = len(self.all_data)
        total_replicas = sum(len(complex_data) for complex_data in self.all_data.values())
        
        report.append(f"Total Complexes Analyzed: {total_complexes}")
        report.append(f"Total Replicas: {total_replicas}")
        report.append("")
        
        total_interactions = 0
        for complex_name, complex_data in self.all_data.items():
            report.append(f"{complex_name}:")
            complex_total = 0
            for replica_name, replica_data in complex_data.items():
                if not replica_data.empty:
                    n_interactions = len(replica_data)
                    n_timepoints = len(replica_data['time_stamp'].unique())
                    n_pairs = len(replica_data[['resid_a', 'resid_b']].drop_duplicates())
                    
                    report.append(f"  {replica_name}: {n_interactions:,} interactions, "
                                f"{n_timepoints} timepoints, {n_pairs} unique pairs")
                    complex_total += n_interactions
            report.append(f"  Total: {complex_total:,} interactions")
            total_interactions += complex_total
            report.append("")
        
        report.append(f"OVERALL TOTAL: {total_interactions:,} interactions")
        report.append("")
        
        # Analysis summary
        report.append("2. ANALYSIS RESULTS")
        report.append("-" * 40)
        report.append("")
        
        report.append("Temporal Dynamics Analysis:")
        report.append("✓ Individual plots for each protein complex system")
        report.append("✓ Smoothed interaction counts over time")
        report.append("✓ Interface stability and size tracking")
        report.append("✓ Interaction diversity patterns")
        report.append("✓ System-specific stability assessments")
        report.append("")
        
        report.append("Interaction Stability Analysis:")
        report.append("✓ Separate 2x2 grid layouts for each complex system")
        report.append("✓ Stability categories: Rare (<10%), Transient (10-50%), Stable (>50%)")
        report.append("✓ Statistical summaries for each replica")
        report.append("✓ Enhanced visualizations with color coding")
        report.append("")
        
        # Files generated
        report.append("3. OUTPUT FILES GENERATED")
        report.append("-" * 40)
        output_files = list(self.output_dir.glob("*.png"))
        for file_path in sorted(output_files):
            report.append(f"• {file_path.name}")
        report.append("")
        
        report.append("=" * 80)
        
        # Save and display report
        report_text = "\n".join(report)
        with open(self.output_dir / "temporal_stability_report.txt", "w") as f:
            f.write(report_text)
        
        print(report_text)
        return report_text

    def run_complete_analysis(self):
        """Run the complete focused analysis pipeline"""
        print("=" * 80)
        print("STARTING TEMPORAL DYNAMICS & STABILITY ANALYSIS")
        print("=" * 80)
        
        # Load all data
        self.load_all_data()
        
        if not self.all_data:
            print("❌ No data found! Please check your data directory.")
            return
        
        print(f"✓ Successfully loaded data for {len(self.all_data)} complexes")
        
        # Run analyses
        try:
            # Temporal dynamics analysis
            print("\n" + "📊 " + "="*60)
            self.analyze_temporal_dynamics()
            
            # Interaction stability analysis
            print("\n" + "🎯 " + "="*60)
            complex_stats = self.plot_interaction_stability()
            
            # Cross-complex comparison (publication-ready)
            if complex_stats and len(complex_stats) > 1:
                print("\n" + "📈 " + "="*60)
                self.plot_cross_complex_comparison(complex_stats)
            
            # Generate summary report
            print("\n" + "📋 " + "="*60)
            self.generate_summary_report()
            
            print("\n" + "✅ " + "="*60)
            print("PUBLICATION-READY ANALYSIS COMPLETE!")
            print(f"📁 All results saved to: {self.output_dir}")
            print("🔬 Generated publication-quality figures in PNG and PDF formats")
            print("📊 Includes grouped stability plots and cross-complex comparisons")
            print("="*80)
            
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main execution function"""
    print("🧬 Temporal Dynamics & Interaction Stability Analysis")
    print("=" * 80)
    
    # Create analysis instance
    analyzer = TemporalStabilityAnalysis(
        data_dir="data", 
        output_dir="temporal_stability_results"
    )
    
    # Run complete analysis
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()
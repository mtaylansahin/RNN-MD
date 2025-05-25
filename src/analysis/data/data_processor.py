"""Data processor for transforming loaded data into analysis-ready format."""

import sys
from pathlib import Path

# Add src to path for absolute imports
sys.path.insert(0, str(Path(__file__).parents[2]))

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .data_loader import LoadedData
from core.utils import get_logger


logger = get_logger(__name__)


@dataclass
class ProcessedData:
    """Container for processed analysis data."""
    
    # Processed datasets
    test_set_processed: pd.DataFrame
    train_set_processed: pd.DataFrame
    valid_set_processed: pd.DataFrame
    output_processed: pd.DataFrame
    
    # Full datasets with interaction grids
    ground_truth_full: pd.DataFrame
    predictions_full: pd.DataFrame
    baseline_full: pd.DataFrame
    
    # Metadata
    all_test_pairs: set
    test_timestamps: List[int]
    total_possible_pairs: int
    
    # Stability analysis
    stability_bins: Optional[pd.Series] = None
    pair_freq_train: Optional[pd.Series] = None


class DataProcessor:
    """Processes loaded data into analysis-ready format."""
    
    def __init__(self):
        """Initialize data processor."""
        self.logger = get_logger(__name__)
    
    def process_all_data(self, data: LoadedData) -> ProcessedData:
        """Process all loaded data into analysis-ready format.
        
        Args:
            data: Loaded raw data
            
        Returns:
            ProcessedData container with processed datasets
        """
        self.logger.info("Starting data processing")
        
        try:
            # Process individual datasets
            test_processed = self._process_dataset(data.test_set, data.labels, include_relation=True)
            train_processed = self._process_dataset(data.train_set, data.labels, include_relation=True)
            valid_processed = self._process_dataset(data.valid_set, data.labels, include_relation=True)
            output_processed = self._process_dataset(data.output, data.labels, include_relation=False)
            
            # Create interaction grids
            test_timestamps = sorted(list(test_processed['time_stamp'].unique()))
            all_test_pairs = self._get_all_relevant_pairs(test_processed, output_processed)
            
            ground_truth_full = self._create_interaction_grid(
                test_processed, all_test_pairs, test_timestamps, "ground_truth"
            )
            predictions_full = self._create_interaction_grid(
                output_processed, all_test_pairs, test_timestamps, "predictions"
            )
            
            # Generate baseline predictions
            baseline_full = self._generate_baseline_predictions(
                train_processed, all_test_pairs, test_timestamps
            )
            
            # Calculate stability bins
            stability_bins, pair_freq_train = self._calculate_stability_bins(train_processed)
            
            # Calculate total possible pairs
            total_possible_pairs = self._calculate_total_possible_pairs(data)
            
            processed_data = ProcessedData(
                test_set_processed=test_processed,
                train_set_processed=train_processed,
                valid_set_processed=valid_processed,
                output_processed=output_processed,
                ground_truth_full=ground_truth_full,
                predictions_full=predictions_full,
                baseline_full=baseline_full,
                all_test_pairs=all_test_pairs,
                test_timestamps=test_timestamps,
                total_possible_pairs=total_possible_pairs,
                stability_bins=stability_bins,
                pair_freq_train=pair_freq_train
            )
            
            self.logger.info("Data processing completed successfully")
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Data processing failed: {e}")
            raise
    
    def _process_dataset(
        self,
        dataset: pd.DataFrame,
        labels: pd.DataFrame,
        include_relation: bool = True
    ) -> pd.DataFrame:
        """Process a dataset by adding name mappings and derived columns.
        
        Args:
            dataset: Dataset to process
            labels: Labels dataframe for name mapping
            include_relation: Whether to include relation information
            
        Returns:
            Processed dataset
        """
        # Create label mapping
        label_map = self._create_label_mapping(labels)
        
        # Add name mappings
        processed = dataset.copy()
        processed['subject_name'] = processed['subject'].map(label_map)
        processed['obj_name'] = processed['object'].map(label_map)
        
        # Create pair identifier
        processed['pair'] = processed['subject_name'] + '_' + processed['obj_name']
        
        # Add relation-specific columns if needed
        if include_relation and 'relation' in processed.columns:
            processed['pair_relation'] = (
                processed['pair'] + '_' + processed['relation'].astype(str)
            )
        
        # Add time-based identifiers
        processed['pair_time'] = processed['pair'] + ' ' + processed['time_stamp'].astype(str)
        
        # Remove rows with missing name mappings
        before_count = len(processed)
        processed = processed.dropna(subset=['subject_name', 'obj_name']).reset_index(drop=True)
        after_count = len(processed)
        
        if before_count > after_count:
            self.logger.warning(
                f"Removed {before_count - after_count} rows with missing name mappings"
            )
        
        return processed
    
    def _create_label_mapping(self, labels: pd.DataFrame) -> Dict[int, str]:
        """Create mapping from label IDs to residue names.
        
        Args:
            labels: Labels dataframe
            
        Returns:
            Dictionary mapping label IDs to residue names
        """
        # Combine residue information from both chains
        label_df = pd.DataFrame()
        label_df['naming'] = list(labels['residue_a']) + list(labels['residue_b'])
        label_df['label'] = list(labels['res_label_a']) + list(labels['res_label_b'])
        label_df = label_df.drop_duplicates().reset_index(drop=True)
        
        # Create mapping dictionary
        label_map = dict(zip(label_df['label'], label_df['naming']))
        return label_map
    
    def _get_all_relevant_pairs(self, test_data: pd.DataFrame, pred_data: pd.DataFrame) -> set:
        """Get all unique pairs from test and prediction data.
        
        Args:
            test_data: Processed test dataset
            pred_data: Processed prediction dataset
            
        Returns:
            Set of all unique pairs
        """
        test_pairs = set(test_data['pair'].unique())
        pred_pairs = set(pred_data['pair'].unique())
        all_pairs = test_pairs | pred_pairs
        
        self.logger.info(f"Found {len(all_pairs)} unique pairs across test and predictions")
        return all_pairs
    
    def _create_interaction_grid(
        self,
        dataset: pd.DataFrame,
        all_pairs: set,
        timestamps: List[int],
        grid_type: str
    ) -> pd.DataFrame:
        """Create full interaction grid for analysis.
        
        Args:
            dataset: Dataset to create grid from
            all_pairs: All unique pairs to include
            timestamps: All timestamps to include
            grid_type: Type of grid being created (for logging)
            
        Returns:
            Full interaction grid dataframe
        """
        # Create complete grid
        grid_data = []
        for pair in all_pairs:
            for timestamp in timestamps:
                grid_data.append({'pair': pair, 'time_stamp': timestamp})
        
        full_grid = pd.DataFrame(grid_data)
        
        # Get observed interactions
        observed = dataset[['pair', 'time_stamp']].drop_duplicates()
        observed['present'] = 1
        
        # Merge with full grid
        full_grid = pd.merge(
            full_grid, observed,
            on=['pair', 'time_stamp'],
            how='left'
        ).fillna({'present': 0})
        
        # Add name information
        pair_names = self._extract_pair_names(dataset)
        full_grid = pd.merge(full_grid, pair_names, on='pair', how='left')
        
        full_grid['present'] = full_grid['present'].astype(int)
        
        self.logger.info(
            f"Created {grid_type} grid: {len(full_grid)} total entries, "
            f"{full_grid['present'].sum()} interactions"
        )
        
        return full_grid
    
    def _extract_pair_names(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Extract unique pair name mappings from dataset.
        
        Args:
            dataset: Dataset with pair name information
            
        Returns:
            Dataframe with unique pair to name mappings
        """
        if 'subject_name' in dataset.columns and 'obj_name' in dataset.columns:
            pair_names = dataset[['pair', 'subject_name', 'obj_name']].drop_duplicates()
        else:
            # Create empty mappings if names not available
            unique_pairs = dataset['pair'].unique()
            pair_names = pd.DataFrame({
                'pair': unique_pairs,
                'subject_name': [pair.split('_')[0] for pair in unique_pairs],
                'obj_name': ['_'.join(pair.split('_')[1:]) for pair in unique_pairs]
            })
        
        return pair_names
    
    def _generate_baseline_predictions(
        self,
        train_data: pd.DataFrame,
        all_test_pairs: set,
        test_timestamps: List[int]
    ) -> pd.DataFrame:
        """Generate baseline predictions using training data.
        
        Args:
            train_data: Training dataset
            all_test_pairs: All test pairs to predict for
            test_timestamps: All test timestamps
            
        Returns:
            Baseline predictions dataframe
        """
        # Get all training interaction pairs
        train_pairs = set(train_data['pair'].unique())
        
        self.logger.info(f"Generating baseline from {len(train_pairs)} training pairs")
        
        # Generate baseline: predict all training pairs at all test times
        baseline_data = []
        for pair in train_pairs:
            for timestamp in test_timestamps:
                baseline_data.append({
                    'pair': pair,
                    'time_stamp': timestamp,
                    'present': 1
                })
        
        baseline_df = pd.DataFrame(baseline_data)
        
        # Create full grid including non-baseline pairs
        full_baseline = self._create_interaction_grid(
            baseline_df, all_test_pairs, test_timestamps, "baseline"
        )
        
        return full_baseline
    
    def _calculate_stability_bins(
        self,
        train_data: pd.DataFrame
    ) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
        """Calculate stability bins based on training frequency.
        
        Args:
            train_data: Training dataset
            
        Returns:
            Tuple of (stability_bins, pair_frequencies)
        """
        try:
            if train_data.empty:
                self.logger.warning("Empty training data, cannot calculate stability bins")
                return None, None
            
            # Calculate pair frequencies
            total_train_timestamps = train_data['time_stamp'].nunique()
            if total_train_timestamps == 0:
                self.logger.warning("No timestamps in training data")
                return None, None
            
            pair_counts = train_data.groupby('pair').size()
            pair_freq = pair_counts / total_train_timestamps
            
            # Define stability bins
            bins = [-0.01, 0.1, 0.5, 1.01]
            labels = ['Rare (<10%)', 'Moderate (10-50%)', 'Stable (>50%)']
            
            stability_bins = pd.cut(pair_freq, bins=bins, labels=labels, right=False)
            
            self.logger.info("Stability bin distribution:")
            for label, count in stability_bins.value_counts().items():
                self.logger.info(f"  {label}: {count} pairs")
            
            return stability_bins, pair_freq
            
        except Exception as e:
            self.logger.error(f"Failed to calculate stability bins: {e}")
            return None, None
    
    def _calculate_total_possible_pairs(self, data: LoadedData) -> int:
        """Calculate total possible subject-object pair combinations.
        
        Args:
            data: Loaded data for calculating unique entities
            
        Returns:
            Total possible pair combinations
        """
        try:
            # Combine all subject and object IDs
            all_subjects = set()
            all_objects = set()
            
            for dataset in [data.test_set, data.train_set, data.valid_set]:
                if 'subject' in dataset.columns:
                    all_subjects.update(dataset['subject'].unique())
                if 'object' in dataset.columns:
                    all_objects.update(dataset['object'].unique())
            
            total_possible = len(all_subjects) * len(all_objects)
            
            self.logger.info(
                f"Calculated total possible pairs: {len(all_subjects)} subjects × "
                f"{len(all_objects)} objects = {total_possible}"
            )
            
            return total_possible
            
        except Exception as e:
            self.logger.error(f"Failed to calculate total possible pairs: {e}")
            return 0 
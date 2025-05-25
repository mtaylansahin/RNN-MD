"""Metrics calculator for performance analysis."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

from ..data.data_processor import ProcessedData
from ...core.utils import get_logger


logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    
    TP: int
    FP: int
    FN: int
    TN: int
    Recall: float
    Precision: float
    TPR: float
    FPR: float
    F1: float
    MCC: float


@dataclass
class MetricsReport:
    """Container for comprehensive metrics analysis."""
    
    baseline_metrics: Optional[PerformanceMetrics]
    model_metrics: PerformanceMetrics
    metrics_by_stability: Dict[str, PerformanceMetrics]
    metrics_over_time: pd.DataFrame
    cumulative_errors: pd.DataFrame


class MetricsCalculator:
    """Calculates various performance metrics and analyses."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.logger = get_logger(__name__)
    
    def calculate_comprehensive_metrics(
        self,
        processed_data: ProcessedData
    ) -> MetricsReport:
        """Calculate comprehensive metrics report.
        
        Args:
            processed_data: Processed analysis data
            
        Returns:
            Complete metrics report
        """
        self.logger.info("Calculating comprehensive metrics")
        
        try:
            # Prepare evaluation series
            gt_eval, pred_eval, baseline_eval = self._prepare_evaluation_series(processed_data)
            
            total_possible_interactions = (
                processed_data.total_possible_pairs * len(processed_data.test_timestamps)
            )
            
            # Calculate baseline metrics
            baseline_metrics = None
            if baseline_eval is not None and not baseline_eval.empty:
                baseline_metrics = self._calculate_metrics(
                    gt_eval, baseline_eval, total_possible_interactions
                )
            
            # Calculate model metrics
            model_metrics = self._calculate_metrics(
                gt_eval, pred_eval, total_possible_interactions
            )
            
            # Calculate metrics by stability
            metrics_by_stability = self._calculate_metrics_by_stability(
                gt_eval, pred_eval, processed_data
            )
            
            # Calculate metrics over time
            metrics_over_time = self._calculate_metrics_over_time(
                gt_eval, pred_eval, processed_data
            )
            
            # Calculate cumulative errors
            cumulative_errors = self._calculate_cumulative_errors(
                gt_eval, pred_eval, processed_data.test_timestamps
            )
            
            report = MetricsReport(
                baseline_metrics=baseline_metrics,
                model_metrics=model_metrics,
                metrics_by_stability=metrics_by_stability,
                metrics_over_time=metrics_over_time,
                cumulative_errors=cumulative_errors
            )
            
            self.logger.info("Comprehensive metrics calculation completed")
            return report
            
        except Exception as e:
            self.logger.error(f"Metrics calculation failed: {e}")
            raise
    
    def _prepare_evaluation_series(
        self,
        processed_data: ProcessedData
    ) -> Tuple[pd.Series, pd.Series, Optional[pd.Series]]:
        """Prepare evaluation series for metrics calculation.
        
        Args:
            processed_data: Processed analysis data
            
        Returns:
            Tuple of (ground_truth_series, predictions_series, baseline_series)
        """
        # Create evaluation series with pair-time index
        gt_eval = processed_data.ground_truth_full.set_index(['pair', 'time_stamp'])['present']
        pred_eval = processed_data.predictions_full.set_index(['pair', 'time_stamp'])['present']
        
        baseline_eval = None
        if not processed_data.baseline_full.empty:
            baseline_eval = processed_data.baseline_full.set_index(['pair', 'time_stamp'])['present']
        
        return gt_eval, pred_eval, baseline_eval
    
    def _calculate_metrics(
        self,
        gt_series: pd.Series,
        pred_series: pd.Series,
        total_possible: int
    ) -> PerformanceMetrics:
        """Calculate standard performance metrics.
        
        Args:
            gt_series: Ground truth series
            pred_series: Predictions series
            total_possible: Total possible interactions
            
        Returns:
            PerformanceMetrics object
        """
        # Align series
        common_index = gt_series.index.intersection(pred_series.index)
        gt_aligned = gt_series[common_index]
        pred_aligned = pred_series[common_index]
        
        # Calculate confusion matrix components
        TP = ((pred_aligned == 1) & (gt_aligned == 1)).sum()
        FP = ((pred_aligned == 1) & (gt_aligned == 0)).sum()
        FN = ((pred_aligned == 0) & (gt_aligned == 1)).sum()
        TN = total_possible - (TP + FP + FN)
        TN = max(0, TN)  # Ensure TN is not negative
        
        # Calculate metrics
        Recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        TPR = Recall
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        F1 = 2 * ((Precision * Recall) / (Precision + Recall)) if (Precision + Recall) > 0 else 0
        
        # Calculate MCC
        mcc_denom = ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** (1/2)
        MCC = (TP * TN - FP * FN) / mcc_denom if mcc_denom > 0 else 0
        
        return PerformanceMetrics(
            TP=int(TP), FP=int(FP), FN=int(FN), TN=int(TN),
            Recall=float(Recall), Precision=float(Precision),
            TPR=float(TPR), FPR=float(FPR), F1=float(F1), MCC=float(MCC)
        )
    
    def _calculate_metrics_by_stability(
        self,
        gt_eval: pd.Series,
        pred_eval: pd.Series,
        processed_data: ProcessedData
    ) -> Dict[str, PerformanceMetrics]:
        """Calculate metrics grouped by stability bins.
        
        Args:
            gt_eval: Ground truth evaluation series
            pred_eval: Predictions evaluation series
            processed_data: Processed data with stability information
            
        Returns:
            Dictionary mapping stability labels to metrics
        """
        metrics_by_bin = {}
        
        if processed_data.stability_bins is None:
            self.logger.warning("No stability bins available, skipping stability analysis")
            return metrics_by_bin
        
        # Get evaluation pairs
        eval_pairs = gt_eval.index.get_level_values(0).unique()
        
        # Align stability bins with evaluation pairs
        stability_bins_aligned = (
            processed_data.stability_bins
            .reindex(eval_pairs)
            .cat.add_categories('Undefined')
            .fillna('Undefined')
        )
        
        for bin_label in stability_bins_aligned.cat.categories:
            pairs_in_bin = stability_bins_aligned[stability_bins_aligned == bin_label].index
            
            if pairs_in_bin.empty:
                continue
            
            # Filter evaluation series to include only pairs in current bin
            gt_bin = gt_eval[gt_eval.index.get_level_values(0).isin(pairs_in_bin)]
            pred_bin = pred_eval[pred_eval.index.get_level_values(0).isin(pairs_in_bin)]
            
            if gt_bin.empty and pred_bin.empty:
                continue
            
            # Calculate metrics for this bin
            # Use a smaller total possible for per-bin analysis
            bin_possible = len(pairs_in_bin) * len(processed_data.test_timestamps)
            
            metrics = self._calculate_metrics(gt_bin, pred_bin, bin_possible)
            metrics_by_bin[bin_label] = metrics
        
        return metrics_by_bin
    
    def _calculate_metrics_over_time(
        self,
        gt_eval: pd.Series,
        pred_eval: pd.Series,
        processed_data: ProcessedData
    ) -> pd.DataFrame:
        """Calculate metrics cumulatively over time.
        
        Args:
            gt_eval: Ground truth evaluation series
            pred_eval: Predictions evaluation series
            processed_data: Processed data with timestamps
            
        Returns:
            DataFrame with metrics over time
        """
        metrics_over_time = defaultdict(list)
        timestamps_sorted = sorted(processed_data.test_timestamps)
        
        # Align indices
        common_index = gt_eval.index.intersection(pred_eval.index)
        gt_aligned = gt_eval[common_index]
        pred_aligned = pred_eval[common_index]
        
        # Get time values from index
        times = gt_aligned.index.get_level_values(1)
        
        for t_idx, t in enumerate(timestamps_sorted):
            # Filter data up to current time
            mask = times <= t
            gt_cumulative = gt_aligned[mask]
            pred_cumulative = pred_aligned[mask]
            
            # Calculate total possible up to this time
            total_possible_cumulative = processed_data.total_possible_pairs * (t_idx + 1)
            
            # Calculate metrics
            metrics = self._calculate_metrics(
                gt_cumulative, pred_cumulative, total_possible_cumulative
            )
            
            # Store metrics
            metrics_over_time['Time'].append(t)
            metrics_over_time['Recall'].append(metrics.Recall)
            metrics_over_time['Precision'].append(metrics.Precision)
            metrics_over_time['F1'].append(metrics.F1)
            metrics_over_time['MCC'].append(metrics.MCC)
            metrics_over_time['TPR'].append(metrics.TPR)
            metrics_over_time['FPR'].append(metrics.FPR)
        
        return pd.DataFrame(metrics_over_time)
    
    def _calculate_cumulative_errors(
        self,
        gt_eval: pd.Series,
        pred_eval: pd.Series,
        timestamps: List[int]
    ) -> pd.DataFrame:
        """Calculate cumulative errors over time.
        
        Args:
            gt_eval: Ground truth evaluation series
            pred_eval: Predictions evaluation series
            timestamps: List of timestamps
            
        Returns:
            DataFrame with cumulative error information
        """
        errors_over_time = defaultdict(list)
        timestamps_sorted = sorted(timestamps)
        
        # Align indices
        common_index = gt_eval.index.intersection(pred_eval.index)
        gt_aligned = gt_eval[common_index]
        pred_aligned = pred_eval[common_index]
        
        # Get time values from index
        times = gt_aligned.index.get_level_values(1)
        
        cumulative_fp = 0
        cumulative_fn = 0
        
        for t in timestamps_sorted:
            # Filter data at current time
            mask_t = times == t
            gt_t = gt_aligned[mask_t]
            pred_t = pred_aligned[mask_t]
            
            fp_t = ((pred_t == 1) & (gt_t == 0)).sum()
            fn_t = ((pred_t == 0) & (gt_t == 1)).sum()
            
            cumulative_fp += fp_t
            cumulative_fn += fn_t
            
            errors_over_time['Time'].append(t)
            errors_over_time['Cumulative_FP'].append(cumulative_fp)
            errors_over_time['Cumulative_FN'].append(cumulative_fn)
            errors_over_time['Cumulative_Errors'].append(cumulative_fp + cumulative_fn)
        
        return pd.DataFrame(errors_over_time)
    
    def calculate_per_edge_f1(
        self,
        processed_data: ProcessedData
    ) -> Optional[pd.DataFrame]:
        """Calculate F1 score for each edge (pair).
        
        Args:
            processed_data: Processed analysis data
            
        Returns:
            DataFrame with per-edge F1 scores or None if calculation fails
        """
        try:
            # Merge ground truth and predictions
            merged = pd.merge(
                processed_data.ground_truth_full.add_suffix('_gt'),
                processed_data.predictions_full.add_suffix('_pred'),
                left_on=['pair_gt', 'time_stamp_gt'],
                right_on=['pair_pred', 'time_stamp_pred'],
                how='inner'
            )
            
            if merged.empty:
                self.logger.warning("No data after merging ground truth and predictions")
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
                self.logger.warning("No per-edge stats calculated")
                return None
            
            f1_df = pd.DataFrame(per_edge_stats).set_index('pair')
            self.logger.info(f"Calculated F1 scores for {len(f1_df)} pairs")
            
            return f1_df
            
        except Exception as e:
            self.logger.error(f"Failed to calculate per-edge F1: {e}")
            return None
    
    def write_metrics_report(
        self,
        report: MetricsReport,
        output_file_path: str
    ) -> None:
        """Write comprehensive metrics report to file.
        
        Args:
            report: Metrics report to write
            output_file_path: Path to output file
        """
        try:
            with open(output_file_path, "w") as f:
                f.write("=== RNN-MD Performance Metrics Report ===\n\n")
                
                # Baseline metrics
                if report.baseline_metrics:
                    f.write("--- BASELINE Performance ---\n")
                    self._write_metrics_to_file(f, report.baseline_metrics, "BASELINE")
                    f.write("\n")
                else:
                    f.write("--- BASELINE Performance ---\n")
                    f.write("No baseline predictions available.\n\n")
                
                # Model metrics
                f.write("--- MODEL Performance (ALL Interactions) ---\n")
                self._write_metrics_to_file(f, report.model_metrics, "MODEL")
                f.write("\n")
                
                # Stability-based metrics
                if report.metrics_by_stability:
                    f.write("--- Performance by Interaction Stability ---\n")
                    for stability_label, metrics in report.metrics_by_stability.items():
                        output_label = stability_label.replace("Moderate", "Uncommon")
                        if stability_label == "Undefined":
                            output_label = "Not in Train"
                        
                        f.write(f"\nMetrics for {output_label} interactions:\n")
                        self._write_metrics_to_file(f, metrics, output_label)
                    f.write("\n")
            
            self.logger.info(f"Metrics report written to {output_file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to write metrics report: {e}")
    
    def _write_metrics_to_file(
        self,
        file_handle,
        metrics: PerformanceMetrics,
        label: str
    ) -> None:
        """Write metrics to file handle.
        
        Args:
            file_handle: Open file handle
            metrics: Metrics to write
            label: Label for the metrics
        """
        file_handle.write(
            f"Recall: {metrics.Recall:.4f}, "
            f"Precision: {metrics.Precision:.4f}, "
            f"TPR: {metrics.TPR:.4f}, "
            f"FPR: {metrics.FPR:.4f}, "
            f"F1: {metrics.F1:.4f}, "
            f"MCC: {metrics.MCC:.4f}\n"
        ) 
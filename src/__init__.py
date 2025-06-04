"""RNN-MD refactored architecture package."""

__version__ = "2.0.0"
__author__ = "RNN-MD Team"
__description__ = "Refactored RNN-MD for protein-protein interaction dynamics prediction"

# Export main analysis functionality
from .analysis import ResultsManager, AnalysisConfig

__all__ = ["ResultsManager", "AnalysisConfig"] 
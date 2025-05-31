"""Analysis module for RNN-MD results processing and visualization."""

from .results_manager import ResultsManager
from .config import AnalysisConfig

# Import visualization components if available
try:
    from .visualization import VisualizationManager, HeatmapPlotter, MetricsPlotter, TrajectoryPlotter
    __all__ = [
        "ResultsManager", 
        "AnalysisConfig",
        "VisualizationManager",
        "HeatmapPlotter",
        "MetricsPlotter", 
        "TrajectoryPlotter"
    ]
except ImportError:
    # Visualization components not available
    __all__ = ["ResultsManager", "AnalysisConfig"] 
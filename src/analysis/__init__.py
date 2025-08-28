"""Analysis module for RNN-MD results processing and visualization."""

from .results_manager import ResultsManager
from .config import AnalysisConfig

# Import visualization components if available
try:
    from .visualization import VisualizationManager, HeatmapPlotter, MetricsPlotter, TrajectoryPlotter
    visualization_available = True
except ImportError:
    visualization_available = False

# Import multi-replica analysis components if available
try:
    from .multi_replica import MultiReplicaAnalyzer, MultiReplicaPlotter, AggregatedMetrics
    multi_replica_available = True
except ImportError:
    multi_replica_available = False

# Build __all__ list based on available components
__all__ = ["ResultsManager", "AnalysisConfig"]

if visualization_available:
    __all__.extend([
        "VisualizationManager",
        "HeatmapPlotter", 
        "MetricsPlotter",
        "TrajectoryPlotter"
    ])

if multi_replica_available:
    __all__.extend([
        "MultiReplicaAnalyzer",
        "MultiReplicaPlotter", 
        "AggregatedMetrics"
    ]) 
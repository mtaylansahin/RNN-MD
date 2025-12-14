#!/usr/bin/env python3
"""Simple wrapper script for combined heatmap analysis.

This script can be run directly from the project root directory.

Example:
    python combined_heatmap_analysis.py --results_dir ./results_all --output combined_heatmaps
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

if __name__ == '__main__':
    try:
        from analysis.visualization.combined_heatmap import main
        sys.exit(main())
    except ImportError as e:
        print(f"Error importing combined heatmap modules: {e}")
        print("Make sure you're running this from the RNN-MD project root directory.")
        sys.exit(1)
    except Exception as e:
        print(f"Error running combined heatmap analysis: {e}")
        sys.exit(1)


#!/usr/bin/env python3
"""Simple wrapper script for combined multi-replica analysis.

This script can be run directly from the project root directory without
worrying about Python path issues.

Example:
    python combined_multi_replica_analysis.py --results_dir ./results_all --output combined_plots
"""

import sys
import os
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

# Import and run the CLI
if __name__ == '__main__':
    try:
        from analysis.multi_replica.combined_analysis import main
        sys.exit(main())
    except ImportError as e:
        print(f"Error importing combined multi-replica analysis modules: {e}")
        print("Make sure you're running this from the RNN-MD project root directory.")
        sys.exit(1)
    except Exception as e:
        print(f"Error running combined multi-replica analysis: {e}")
        sys.exit(1)


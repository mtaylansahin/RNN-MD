#!/usr/bin/env python3
"""Simple wrapper script for multi-replica analysis.

This script can be run directly from the project root directory without
worrying about Python path issues.
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
        from analysis.multi_replica.cli import main
        sys.exit(main())
    except ImportError as e:
        print(f"Error importing multi-replica analysis modules: {e}")
        print("Make sure you're running this from the RNN-MD project root directory.")
        sys.exit(1)
    except Exception as e:
        print(f"Error running multi-replica analysis: {e}")
        sys.exit(1)
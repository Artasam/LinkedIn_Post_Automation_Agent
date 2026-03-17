"""
main.py
-------
Convenience entry point. Delegates to scheduler/run_pipeline.py.

Usage:
    python main.py              # Live mode
    python main.py --dry-run    # Generate post, skip publishing
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from scheduler.run_pipeline import main

if __name__ == "__main__":
    main()

import os
import sys

# export_pt2.py does `from exportlib import ...`, resolved via the script's own directory
# being on sys.path when run directly (`python scripts/export_pt2.py`). Reproduce that here so
# it can be imported as a module under pytest.
SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

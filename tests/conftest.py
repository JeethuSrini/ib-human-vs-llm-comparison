import sys
from pathlib import Path

# Ensure repo root is on path when running pytest from subdirs
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

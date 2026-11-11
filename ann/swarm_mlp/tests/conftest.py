"""Los módulos de swarm_mlp se importan como top-level (`core`, `swarm`, `runner`),
igual que cuando se corre `python runner.py` desde esta carpeta."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

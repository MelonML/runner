import os
from pathlib import Path

BASE_PATH = Path(os.path.realpath(__file__)).parents[1]

DATA_PATH = BASE_PATH / 'data'
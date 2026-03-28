__version__ = "0.1.0"

from .data_loader import (
    load_subtask,
    StanceDataset,
    normalize_arabic,
    SUBTASK_CONFIG,
    DATA_DIR,
)
from .config import TrainConfig

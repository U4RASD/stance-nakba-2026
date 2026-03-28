"""
Training configuration for stance detection.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainConfig:
    # Task
    subtask: str = "B"  # "A" or "B"
    
    # Model
    model_name: str = "aubmindlab/bert-base-arabertv02"
    max_length: int = 512
    
    # Training
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # Paths
    output_dir: str = "./outputs/stance_model"
    predictions_csv: str = "prediction.csv"
    predictions_zip: str = "prediction.zip"
    
    # Regularisation
    classifier_dropout: Optional[float] = None  # None = use model default

    # Misc
    seed: int = 42
    normalize_arabic: bool = True
    save_checkpoints: bool = True  # False = skip intermediate saves (faster training)
    

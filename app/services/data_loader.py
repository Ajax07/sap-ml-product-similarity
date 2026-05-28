from pathlib import Path

import pandas as pd
from loguru import logger


class DataLoader:
    """Loads Amazon fashion dataset."""

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)

    def load_data(self) -> pd.DataFrame:
        """Load LDJSON dataset."""

        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")

        logger.info(f"Loading dataset from {self.data_path}")

        df = pd.read_json(self.data_path, lines=True)

        logger.info(f"Loaded dataset with shape: {df.shape}")

        return df

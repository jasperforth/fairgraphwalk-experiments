import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Final
from pathlib import Path


class DataFormatter(ABC):
    @abstractmethod
    def pre_formatting(self, data_filepath: str, formatted_data_dir: Path, filter_category: str, 
                        column_names: List[str], attributes: List[str]) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def formatting(self) -> pd.DataFrame:
        pass
    



import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Any
import numpy as np
import h5py
from pathlib import Path


class LoLDataset(Dataset):
    
    def __init__(self, data_path: str, augment: bool = False):
        self.data_path = Path(data_path)
        self.augment = augment
        self.data = self.load_data()
    
    def load_data(self) -> List[Tuple[Any, Any]]:
        pass
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.data[idx]
        
        if self.augment:
            item = self.augment_data(item)
        
        return self._process_item(item)
    
    def augment_data(self, item: Tuple[Any, Any]) -> Tuple[Any, Any]:
        pass
    
    def _process_item(self, item: Tuple[Any, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


def get_dataloader(data_path: str, batch_size: int = 16, 
                    shuffle: bool = True, num_workers: int = 2, 
                    augment: bool = False) -> DataLoader:
    dataset = LoLDataset(data_path, augment=augment)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return dataloader

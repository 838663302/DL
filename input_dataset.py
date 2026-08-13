from torch.utils.data import Dataset, DataLoader
import os
import torch
import pandas as pd
import config
from torch.utils.data.distributed import DistributedSampler

class InputDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_json(file_path, lines=True, orient='records')
        self.input = self.data["input"].tolist()
        self.target = self.data["target"].tolist()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.input[idx], dtype=torch.long), torch.tensor(self.target[idx], dtype=torch.long)



def get_dataloader(batch_size, shuffle, is_train=True, rank=None, world_size=None, num_workers=None):
    if num_workers is None:
        # Windows 下多进程 DataLoader worker 需要 __main__ 保护，默认 0；
        # Linux/Kaggle 下用 2 个子进程并行加载和 padding，缓解 CPU 瓶颈
        num_workers = 0 if os.name == "nt" else 2
    # num_workers>0 时启用常驻 worker 与预取：worker 不随 epoch 销毁重建，
    # 每个 worker 预取 2 个 batch，隐藏加载延迟
    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs = dict(persistent_workers=True, prefetch_factor=2)
    if is_train:
        file_path = config.DATASET_DIR / "iwslt_train_tokenized_input.jsonl"
    else:
        file_path = config.DATASET_DIR / "iwslt_test_tokenized_input.jsonl"
    dataset = InputDataset(file_path)
    # 单进程/推理场景 rank 为空时退化为普通 shuffle
    # input 定长、target 标量，默认 collate 即可，无需 padding
    if rank is not None and world_size is not None:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                          num_workers=num_workers, **loader_kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, **loader_kwargs)
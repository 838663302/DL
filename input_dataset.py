from torch.utils.data import Dataset, DataLoader
import os
import torch
import pandas as pd
import config
from torch.utils.data.distributed import DistributedSampler

class InputDataset(Dataset):
    def __init__(self, file_path):
        data = pd.read_json(file_path, lines=True, orient='records')
        self.length = len(data)
        # 一次性把整列转成预分配大张量：__getitem__ 变纯切片，
        # 避免每条样本都新建 tensor / 走 Python 循环（200 万条时开销巨大）
        self.input = torch.tensor(data["input"].tolist(), dtype=torch.long)      # (N, W)
        self.target = torch.tensor(data["target"].tolist(), dtype=torch.long)    # (N,)

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return self.input[idx], self.target[idx]



def get_dataloader(batch_size, shuffle, is_train=True, rank=None, world_size=None, num_workers=None):
    if num_workers is None:
        # 数据集已整体预载入内存张量，worker 只做切片+collate，本身很轻。
        # 关键是不能开太多：Kaggle 等环境 CPU 核少，DDP 下进程总数 = ranks*(workers+1)，
        # 开多了进程调度互相争抢反而拖慢训练。按核数/进程数自适应分配。
        if os.name == "nt":
            # Windows 下多进程 DataLoader worker 需要 __main__ 保护，默认 0；
            num_workers = 0
        else:
            total_cores = os.cpu_count() or 2
            if world_size:
                num_workers = max(1, min(2, total_cores // world_size))
            else:
                num_workers = max(1, min(2, total_cores // 2))
    # num_workers>0 时启用常驻 worker 与预取：worker 不随 epoch 销毁重建，
    # 每个 worker 预取 2 个 batch，隐藏加载延迟
    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs = dict(persistent_workers=True, prefetch_factor=2)
        if torch.cuda.is_available():
            loader_kwargs["pin_memory"] = True
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
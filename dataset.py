import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

import config

class MyDataSet(Dataset):
    def __init__(self, file_path):
        data = pd.read_json(file_path, lines=True, orient='records')
        # 加载时一次性转成大张量，训练时__getitem__只做切片，避免逐条转换吃满CPU
        self.inputs = torch.tensor(data["input"].tolist(), dtype=torch.long)
        self.targets = torch.tensor(data["target"].tolist(), dtype=torch.long)
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]



class InMemoryLoader:
    """数据常驻GPU的轻量loader：每个epoch打乱索引后直接切片，无DataLoader的逐条取数/拼接/拷贝开销"""
    def __init__(self, inputs, targets, batch_size, shuffle=True):
        self.inputs = inputs
        self.targets = targets
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return (len(self.targets) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.targets)
        if self.shuffle:
            idx = torch.randperm(n, device=self.inputs.device)
        else:
            idx = torch.arange(n, device=self.inputs.device)
        for s in range(0, n, self.batch_size):
            j = idx[s:s + self.batch_size]
            yield self.inputs[j], self.targets[j]


def getLoader(isTrain=True, device=None):
    dataset = MyDataSet(config.DATASET_DIR / (f"train_data_set.jsonl" if isTrain else "test_data_set.jsonl"))
    inputs, targets = dataset.inputs, dataset.targets
    # 整个数据集一次性搬到GPU常驻（训练集约140MB，显存完全装得下），训练循环零CPU参与
    if device is not None:
        inputs, targets = inputs.to(device), targets.to(device)
    return InMemoryLoader(inputs, targets, config.BATCH_SIZE, shuffle=isTrain)

if __name__ == "__main__":
    train_loader = getLoader(True)
    for batch, target in train_loader:
        print(batch, target)
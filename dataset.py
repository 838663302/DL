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



def getLoader(isTrain=True):
    dataset = MyDataSet(config.DATASET_DIR / (f"train_data_set.jsonl" if isTrain else "test_data_set.jsonl"))
    # 数据已全部在内存张量中，无IO开销，num_workers=0避免子进程通信反而拖慢
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                      num_workers=0, pin_memory=torch.cuda.is_available())

if __name__ == "__main__":
    train_loader = getLoader(True)
    for batch, target in train_loader:
        print(batch, target)
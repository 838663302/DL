import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

import config

class MyDataSet(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_json(file_path, lines=True, orient='records').to_dict(orient='records')
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        input = self.data[index]["input"]
        target = self.data[index]["target"]
        return torch.tensor(input, dtype=torch.long), torch.tensor(target, dtype=torch.long)



def getLoader(isTrain=True):
    dataset = MyDataSet(config.DATASET_DIR / (f"train_data_set.jsonl" if isTrain else "test_data_set.jsonl"))
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

if __name__ == "__main__":
    train_loader = getLoader(True)
    for batch, target in train_loader:
        print(batch, target)
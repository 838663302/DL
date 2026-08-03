from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import config
from tokenizer import ZHTokenizer, ENTokenizer


class Seq2SeqDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_json(file_path, lines=True, orient='records')
        self.input = self.data["zh"].tolist()
        self.target = self.data["en"].tolist()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.input[idx], dtype=torch.long), torch.tensor(self.target[idx], dtype=torch.long)

def pad_collate(batch):
    input_batch, target_batch = zip(*batch)
    zh_tokenizer = ZHTokenizer.from_vocab(config.ZH_VOCAB_PATH)
    en_tokenizer = ENTokenizer.from_vocab(config.EN_VOCAB_PATH)
    input_batch = torch.nn.utils.rnn.pad_sequence(input_batch, batch_first=True, padding_value=zh_tokenizer.pad_id)
    target_batch = torch.nn.utils.rnn.pad_sequence(target_batch, batch_first=True, padding_value=en_tokenizer.pad_id)
    return input_batch, target_batch


def get_dataloader(batch_size, shuffle, is_train=True):
    if is_train:
        file_path = config.DATASET_DIR / "iwslt_train_tokenized.jsonl"
    else:
        file_path = config.DATASET_DIR / "iwslt_test_tokenized.jsonl"
    return DataLoader(Seq2SeqDataset(file_path), batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate)

if __name__ == "__main__":
    dataloader = get_dataloader(batch_size=config.BATCH_SIZE, shuffle=True, is_train=False)
    for batch, target in dataloader:
        print(batch, target)

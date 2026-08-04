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
    # 词表全局缓存：只在首次加载，避免每个batch都重读文件
_zh_tokenizer = None
_en_tokenizer = None

def _get_tokenizers():
    global _zh_tokenizer, _en_tokenizer
    if _zh_tokenizer is None:
        _zh_tokenizer = ZHTokenizer.from_vocab(config.ZH_VOCAB_PATH)
        _en_tokenizer = ENTokenizer.from_vocab(config.EN_VOCAB_PATH)
    return _zh_tokenizer, _en_tokenizer

def pad_collate(batch):
    input_batch, target_batch = zip(*batch)
    zh_tokenizer, en_tokenizer = _get_tokenizers()
    # 截断到MAX_SEQ_LEN，防止batch内超长序列把padding长度撑爆导致显存溢出
    # zh截到MAX_SEQ_LEN；en带sos/eos，截到MAX_SEQ_LEN+1保证切分后两边都≤MAX_SEQ_LEN
    input_batch = [seq[:config.MAX_SEQ_LEN] for seq in input_batch]
    target_batch = [seq[:config.MAX_SEQ_LEN + 1] for seq in target_batch]
    input_batch = torch.nn.utils.rnn.pad_sequence(input_batch, batch_first=True, padding_value=zh_tokenizer.pad_id)
    target_batch = torch.nn.utils.rnn.pad_sequence(target_batch, batch_first=True, padding_value=en_tokenizer.pad_id)
    # 统一长度以便训练时 stack 成 (batch, 2, seq) 打包输入：
    # input pad到max_len，target pad到max_len+1，这样 target[:, :-1] 与 input 等长
    max_len = max(input_batch.size(1), target_batch.size(1) - 1)
    if input_batch.size(1) < max_len:
        input_batch = torch.nn.functional.pad(input_batch, (0, max_len - input_batch.size(1)), value=zh_tokenizer.pad_id)
    if target_batch.size(1) < max_len + 1:
        target_batch = torch.nn.functional.pad(target_batch, (0, max_len + 1 - target_batch.size(1)), value=en_tokenizer.pad_id)
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

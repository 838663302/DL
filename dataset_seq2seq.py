from torch.utils.data import Dataset, DataLoader
import os
import torch
import pandas as pd
import config
from tokenizer import ZHTokenizer, ENTokenizer
from torch.utils.data.distributed  import DistributedSampler


class Seq2SeqDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_json(file_path, lines=True, orient='records')
        self.input = self.data["zh"].tolist()
        self.target = self.data["en"].tolist()
        # 预计算每条样本的截断后长度（与 pad_collate 的截断规则一致），
        # 供 BucketDistributedSampler 按长度分桶，减少 batch 内 padding 浪费
        self.lengths = [
            max(min(len(z), config.MAX_SEQ_LEN), min(len(t), config.MAX_SEQ_LEN + 1))
            for z, t in zip(self.input, self.target)
        ]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.input[idx], dtype=torch.long), torch.tensor(self.target[idx], dtype=torch.long)


class BucketDistributedSampler(DistributedSampler):
    """在 DistributedSampler 分片基础上按长度分桶打乱：
    每个 rank 先拿到不重叠的索引子集，再按长度排序切成块，
    同块内长度接近，DataLoader 顺序取 batch 时 padding 大幅减少。
    块顺序和块内顺序用 self.epoch 作随机种子，保证每个 epoch 数据顺序不同。
    """
    def __init__(self, dataset, num_replicas, rank, shuffle=True,
                 seed=0, drop_last=False, bucket_block_size=256):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank,
                         shuffle=shuffle, seed=seed, drop_last=drop_last)
        self.lengths = getattr(dataset, "lengths", None)
        self.bucket_block_size = bucket_block_size

    def __iter__(self):
        if not self.shuffle or self.lengths is None:
            return super().__iter__()

        # 先取本 rank 的数据子集（保证各卡不重叠），再按长度排序
        indices = list(super().__iter__())
        indices.sort(key=lambda i: self.lengths[i])
        # 切成固定大小的块，块内长度接近
        blocks = [indices[i:i + self.bucket_block_size]
                  for i in range(0, len(indices), self.bucket_block_size)]
        g = torch.Generator()
        g.manual_seed(self.epoch)
        block_order = torch.randperm(len(blocks), generator=g).tolist()
        result = []
        for bi in block_order:
            b = blocks[bi]
            perm = torch.randperm(len(b), generator=g).tolist()
            result.extend(b[p] for p in perm)
        return iter(result)

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
    # 注意：input 与 target 各自按 batch 内最长截断，无需强制等长。
    # 编码器只依赖 src 长度，解码器只依赖 tgt 长度，二者完全解耦（DDP 改造后）
    return input_batch, target_batch


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
        file_path = config.DATASET_DIR / "iwslt_train_tokenized.jsonl"
    else:
        file_path = config.DATASET_DIR / "iwslt_test_tokenized.jsonl"
    dataset = Seq2SeqDataset(file_path)
    # DDP 下用 BucketDistributedSampler：分片 + 按长度分桶打乱，减少 batch 内 padding；
    # 单进程/推理场景 rank 为空时退化为普通 shuffle
    if rank is not None and world_size is not None:
        sampler = BucketDistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=shuffle,
            bucket_block_size=batch_size * 4)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                          collate_fn=pad_collate, num_workers=num_workers, **loader_kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=pad_collate, num_workers=num_workers, **loader_kwargs)

if __name__ == "__main__":
    dataloader = get_dataloader(batch_size=config.BATCH_SIZE, shuffle=True, is_train=False)
    for batch, target in dataloader:
        print(batch, target)

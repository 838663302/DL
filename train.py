import os
# 必须在import torch之前设置：启用CUDA显存扩展段，减少碎片化导致的OOM
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import config
from model import Translator
from dataset_seq2seq import get_dataloader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tokenizer import ZHTokenizer, ENTokenizer


def train(rank, world_size):
    device = torch.device(f"cuda:{rank}")
    zhTokenizer = ZHTokenizer.from_vocab(config.ZH_VOCAB_PATH)
    enTokenizer = ENTokenizer.from_vocab(config.EN_VOCAB_PATH)
    dataloader = get_dataloader(
        batch_size=config.BATCH_SIZE, shuffle=True, is_train=True,
        rank=rank, world_size=world_size
    )
    model = Translator(
        zh_vocab_size=zhTokenizer.vocab_size,
        en_vocab_size=enTokenizer.vocab_size,
        d_model=config.EMBEDDING_DIM,
        zh_pad_id=zhTokenizer.pad_id,
        en_pad_id=enTokenizer.pad_id
    ).to(device)
    model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    criterion = nn.CrossEntropyLoss(ignore_index=enTokenizer.pad_id)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.94)
    loss_value = float("inf")

    for epoch in range(config.EPOCHS):
        # 每个 epoch 重新打乱数据，保证各卡样本分配随 epoch 变化
        dataloader.sampler.set_epoch(epoch)
        loss_batch = train_one_epoch(model, dataloader, optimizer, criterion, enTokenizer, device)
        if rank == 0:
            print(f"Epoch {epoch+1}, Loss: {loss_batch}")
        scheduler.step()

        if loss_value > loss_batch:
            loss_value = loss_batch
            if rank == 0:
                # DDP 包装后取内部模型保存，避免保存带 module. 前缀的状态字典
                torch.save(model.module.state_dict(), os.path.join(config.CHECKPOINT_DIR, "best_model.pth"))

def train_one_epoch(model, dataloader, optimizer, criterion, enTokenizer, device):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch, target in dataloader:
        batch = batch.to(device)
        target = target.to(device)
        input_targets = target[:, :-1]
        output_targets = target[:, 1:]

        output = model(batch, input_targets)
        # output shape: (batch_size, seq_len, en_vocab_size)
        # output_targets shape: (batch_size, seq_len)
        loss = criterion(output.reshape(-1, output.size(-1)), output_targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else float('inf')

def predict_batch(input_batch, model, zhTokenizer, enTokenizer):
    # 推理函数独立于 DDP 训练使用，自行推断设备
    device = input_batch.device
    model.eval()
    # 创建形状为 (batch_size, 1) 的全零张量，用作解码器初始输入（SOS token）
    input_target = torch.full((input_batch.shape[0], 1), fill_value=enTokenizer.sos_id, dtype=torch.long).to(device)
    src_padding_mask = (input_batch == zhTokenizer.pad_id).to(device)
    generated = []
    is_all = torch.full((input_batch.shape[0],), fill_value=False, dtype=torch.bool).to(device)
    with torch.no_grad():
        for i in range(config.MAX_SEQ_LEN):
            #  def encode(self, src, src_padding_mask):
            # memory size: (batch_size, seq_len, d_model)
            memory = model.encode(input_batch, src_padding_mask)
            #def decode(self, tgt, tgt_mask, memory, memory_key_padding_mask):
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(input_target.size(1)).to(device)
            memory_key_padding_mask = (input_batch == zhTokenizer.pad_id).to(device)
            # decode_output size: (batch_size, seq_len, en_vocab_size)
            decode_output = model.decode(input_target, tgt_mask, memory, memory_key_padding_mask)
            decode_output = decode_output[:, -1, :]
            decode_output = decode_output.argmax(dim=-1) # (batch_size, )
            generated.append(decode_output.unsqueeze(1))
            input_target = torch.cat((input_target, decode_output.unsqueeze(1)), dim=1)
            is_all |= decode_output.eq(enTokenizer.eos_id)
            if torch.all(is_all):
                break

    generated = torch.cat(generated, dim=1).to(device)
    return generated

def init_process(rank, world_size, fn):
    """每个子进程的入口：初始化进程组、设置设备、运行训练、销毁进程组"""
    torch.cuda.set_device(rank)
    # env:// 方式通过 mp.spawn 注入的 RANK/WORLD_SIZE 等环境变量建立进程组
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    fn(rank, world_size)
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    print(f"使用 {world_size} 张 GPU 进行 DDP 训练")
    mp.spawn(init_process, args=(world_size, train), nprocs=world_size)
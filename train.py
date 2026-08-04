import os
# 必须在import torch之前设置：启用CUDA显存扩展段，减少碎片化导致的OOM
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import config
from model import Translator
from dataset_seq2seq import get_dataloader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tokenizer import ZHTokenizer, ENTokenizer

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"训练设备: {device}")

def train():
    zhTokenizer = ZHTokenizer.from_vocab(config.ZH_VOCAB_PATH)
    enTokenizer = ENTokenizer.from_vocab(config.EN_VOCAB_PATH)
    dataloader = get_dataloader(batch_size=config.BATCH_SIZE, shuffle=True, is_train=True)
    model = Translator(
        zh_vocab_size=zhTokenizer.vocab_size,
        en_vocab_size=enTokenizer.vocab_size,
        d_model=config.EMBEDDING_DIM,
        zh_pad_id=zhTokenizer.pad_id,
        en_pad_id=enTokenizer.pad_id
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    criterion = nn.CrossEntropyLoss(ignore_index=enTokenizer.pad_id)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.94)
    loss_value = float("inf")
    
    for epoch in range(config.EPOCHS):
        loss_batch = train_one_epoch(model, dataloader, optimizer, criterion, enTokenizer)
        print(f"Epoch {epoch+1}, Loss: {loss_batch}")
        scheduler.step()

        if loss_value > loss_batch:
            loss_value = loss_batch
            torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, "best_model.pth"))

def train_one_epoch(model, dataloader, optimizer, criterion, enTokenizer):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch, target in dataloader:
        batch = batch.to(device)
        target = target.to(device)
        input_targets = target[:, :-1]
        output_targets = target[:, 1:]
        
        # DataParallel只切分第一个参数，所以把src/tgt打包成单个张量 (batch, 2, seq)，
        # 切分后每卡拿到 (batch/2, 2, seq)，mask在模型内部基于切分后的输入生成
        combined = torch.stack([batch, input_targets], dim=1)
        
        output = model(combined)
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
    
if __name__ == "__main__":
    train()
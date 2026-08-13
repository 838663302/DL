import os
import subprocess
# 必须在import torch之前设置：启用CUDA显存扩展段，减少碎片化导致的OOM
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
# Kaggle 等云环境 T4 之间 NCCL P2P 通信可能卡死，禁用 P2P 走更稳妥的传输路径
os.environ["NCCL_P2P_DISABLE"] = "1"

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
    # Noam 调度：先线性 warmup 上升，再按 1/sqrt(step) 缓慢衰减。
    # 相比 ExponentialLR 每 epoch 乘 0.94（后期学习率太小导致 loss 停滞），
    # 后期仍有足够学习能力，更适合大模型长时间训练
    warmup_steps = config.WARMUP_STEPS
    def lr_lambda(step):
        step = max(step, 1)
        return min(step / warmup_steps, 1.0) * (warmup_steps ** 0.5) * (step ** -0.5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    loss_value = float("inf")

    # 只在 rank=0 创建 TensorBoard writer，避免两个进程重复写日志目录
    writer = SummaryWriter(log_dir=config.OUTPUT_DIR / "runs") if rank == 0 else None

    for epoch in range(config.EPOCHS):
        # 每个 epoch 重新打乱数据，保证各卡样本分配随 epoch 变化
        dataloader.sampler.set_epoch(epoch)
        loss_batch = train_one_epoch(model, dataloader, optimizer, criterion, enTokenizer, device, scheduler)
        if rank == 0:
            print(f"Epoch {epoch+1}, Loss: {loss_batch}")
            # 记录每个 epoch 的平均损失和学习率
            writer.add_scalar("loss/train", loss_batch, epoch)
            writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

        if loss_value > loss_batch:
            loss_value = loss_batch
            if rank == 0:
                # DDP 包装后取内部模型保存，避免保存带 module. 前缀的状态字典
                torch.save(model.module.state_dict(), os.path.join(config.CHECKPOINT_DIR, "best_model.pth"))

    if writer is not None:
        writer.close()

def train_one_epoch(model, dataloader, optimizer, criterion, enTokenizer, device, scheduler=None):
    model.train()
    total_loss = 0.0
    num_batches = 0
    # 每多少步打印一次 GPU 利用率（观察 CPU/GPU 瓶颈用，仅在 rank0 打印避免刷屏）
    gpu_monitor_interval = 200

    for step, (batch, target) in enumerate(dataloader):
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
        # Noam 调度按 batch 步数更新
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        # if step % gpu_monitor_interval == 0 and device.index == 0:
        #     try:
        #         util = subprocess.check_output(
        #             "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader",
        #             shell=True).decode().strip().replace("\n", " | ")
        #     except Exception:
        #         util = "nvidia-smi 不可用"
        #     print(f"  step {step}: {util}")

    return total_loss / num_batches if num_batches > 0 else float('inf')

def beam_search_decode(model, memory, src_padding_mask, enTokenizer, device, max_len):
    """批量束搜索。memory: (B, seq_len, d_model)，flat index = b*K + k（第 b 句第 k 束）"""
    B = memory.shape[0]
    K = config.BEAM_SIZE
    alpha = config.LENGTH_PENALTY_ALPHA
    pad_id, sos_id, eos_id = enTokenizer.pad_id, enTokenizer.sos_id, enTokenizer.eos_id

    memory_flat = memory.repeat_interleave(K, dim=0)      # (B*K, seq_len, d_model)
    mask_flat = src_padding_mask.repeat_interleave(K, dim=0)  # (B*K, seq_len)

    seqs = torch.full((B * K, 1), sos_id, dtype=torch.long, device=device)   # 初始全为 [sos]
    cum_log_prob = torch.zeros(B * K, device=device)
    active = torch.ones(B * K, dtype=torch.bool, device=device)  # True=尚未出 EOS，可继续扩展

    finished = [[] for _ in range(B)]   # 每句的 (tokens(不含sos/eos), log_prob)

    with torch.no_grad():
        for step in range(max_len):
            if not active.any():
                break
            cur_len = seqs.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(cur_len).to(device)
            logits = model.decode(seqs, tgt_mask, memory_flat, mask_flat)  # (B*K, cur_len, vocab)
            log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)        # (B*K, vocab)

            # 扩展 active 束的 topK，得 (B*K, K) 个新 token 与增量 logp
            top_logp, top_idx = log_probs.topk(K, dim=-1)   # (B*K, K)

            new_seqs = []      # (b, cands)，cands 是第 b 句某束扩展出的 K 个候选
            for flat in range(B * K):
                if not active[flat]:
                    continue
                b, k = divmod(flat, K)
                row_tokens = seqs[flat].tolist()
                cands = []
                for j in range(K):
                    cands.append((row_tokens + [top_idx[flat, j].item()],
                                  cum_log_prob[flat].item() + top_logp[flat, j].item()))
                new_seqs.append((b, cands))

            next_seqs = torch.full((B * K, cur_len + 1), pad_id, dtype=torch.long, device=device)
            next_logp = torch.full((B * K,), -1e9, device=device)
            next_active = torch.zeros(B * K, dtype=torch.bool, device=device)

            for b in range(B):
                # 汇总第 b 句所有 active 束扩展出的候选
                pool = []
                for bb, cands in new_seqs:
                    if bb == b:
                        pool.extend(cands)
                # 按累积 logp 降序
                pool.sort(key=lambda x: x[1], reverse=True)

                # 非 EOS 取前 K 个
                slot = 0
                for tokens, logp in pool:
                    if tokens[-1] == eos_id:
                        finished[b].append((tokens[1:-1], logp))
                    else:
                        if slot >= K:
                            break
                        if tokens[-1] != eos_id:
                            flat = b * K + slot
                            next_seqs[flat, :cur_len + 1] = torch.tensor(tokens, dtype=torch.long, device=device)
                            next_logp[flat] = logp
                            next_active[flat] = True
                            slot += 1

            seqs, cum_log_prob, active = next_seqs, next_logp, next_active

    # 逐句长度归一化（Google NMT），挑每句最优序列
    def length_norm(tokens):
        return (6 ** alpha) / ((5 + len(tokens)) ** alpha)

    results = []
    for b in range(B):
        # 未完成束：只用 active 的真实束（占位束 active=False 被排除；-1e9 初始分数也是双保险）
        unfinished = [(seqs[b*K+k, 1:].tolist(), cum_log_prob[b*K+k].item())
                      for k in range(K)
                      if active[b*K+k] and cum_log_prob[b*K+k].item() > -1e6]
        cands = finished[b] + unfinished
        cands = [c for c in cands if len(c[0]) > 0]
        if not cands:
            results.append([eos_id])
            continue
        best = max(cands, key=lambda x: x[1] * length_norm(x[0]))
        results.append(best[0])
    return results  # list[list]，每句一个 token 序列

def predict_batch(input_batch, model, zhTokenizer, enTokenizer):
    # 推理函数独立于 DDP 训练使用，自行推断设备
    device = input_batch.device
    model.eval()
    src_padding_mask = (input_batch == zhTokenizer.pad_id).to(device)
    # 编码器只算一次：输入不变，束搜索内部循环无需重复编码
    # memory size: (batch_size, seq_len, d_model)
    memory = model.encode(input_batch, src_padding_mask)
    results = beam_search_decode(model, memory, src_padding_mask,
                                 enTokenizer, device, config.MAX_SEQ_LEN)
    # 用 pad_sequence 把变长序列列表 pad 成 (batch, seq_len) 张量
    tensors = [torch.tensor(seq, dtype=torch.long, device=device) for seq in results]
    if tensors:
        padded = torch.nn.utils.rnn.pad_sequence(
            tensors, batch_first=True, padding_value=enTokenizer.pad_id)
    else:
        padded = torch.empty((0, 0), dtype=torch.long, device=device)
    return padded

def predict():
    # 有 GPU 用 GPU，否则回退 CPU（本地调试可用）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    zhTokenizer = ZHTokenizer.from_vocab(config.ZH_VOCAB_PATH)
    enTokenizer = ENTokenizer.from_vocab(config.EN_VOCAB_PATH)
    model = Translator(
        zh_vocab_size=zhTokenizer.vocab_size,
        en_vocab_size=enTokenizer.vocab_size,
        d_model=config.EMBEDDING_DIM,
        zh_pad_id=zhTokenizer.pad_id,
        en_pad_id=enTokenizer.pad_id
    ).to(device)
    model.load_state_dict(torch.load(
        os.path.join(config.CHECKPOINT_DIR, "best_model.pth"), map_location=device))
    model.eval()
    while True:
        zh_sentence = input("请输入中文句子（输入 exit 退出）：")
        if zh_sentence == "exit":
            break
        zh_ids = zhTokenizer.encode(zh_sentence)
        # 超长输入截断，避免位置编码越界
        zh_ids = zh_ids[:config.MAX_SEQ_LEN]
        zh_ids = torch.tensor(zh_ids, dtype=torch.long).unsqueeze(0).to(device)
        en_ids = predict_batch(zh_ids, model, zhTokenizer, enTokenizer)
        en_ids_list = en_ids.tolist()[0]
        # 截断到 eos 标记，避免输出尾部出现 <eos>
        if enTokenizer.eos_id in en_ids_list:
            en_ids_list = en_ids_list[:en_ids_list.index(enTokenizer.eos_id)]
        en_sentence = enTokenizer.decode(en_ids_list)
        # 去掉生僻词兜底的 <unk>，日常使用不展示
        en_sentence = en_sentence.replace(enTokenizer.oov, "")
        print(f"翻译结果：{en_sentence}")

def init_process(rank, world_size, fn):
    """每个子进程的入口：初始化进程组、设置设备、运行训练、销毁进程组"""
    torch.cuda.set_device(rank)
    # mp.spawn 不会注入 MASTER_ADDR/MASTER_PORT，必须手动指定（单机多卡用回环地址）
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    fn(rank, world_size)
    dist.destroy_process_group()

if __name__ == "__main__":
    # world_size = 2
    # print(f"使用 {world_size} 张 GPU 进行 DDP 训练")
    # mp.spawn(init_process, args=(world_size, train), nprocs=world_size)
    predict()
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
from model import InputMethod
from input_dataset import get_dataloader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tokenizer import ZHTokenizer


def load_checkpoint(device=None, ckpt_name="best_model.pth"):
    """读取 checkpoint 文件，只返回权重和断点数值，不构建/加载模型。

    返回 dict 或 None：
        None     : 文件不存在
        完整格式 : {
            "format": "full", model, optimizer, scheduler, scaler,
            start_epoch, best_loss
        }
        旧格式   : {
            "format": "old", model(纯 state_dict),
            optimizer=None, scheduler=None, scaler=None,
            start_epoch=0, best_loss=inf
        }
    模型构建与状态恢复由调用方完成。
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resume_path = os.path.join(config.CHECKPOINT_DIR, ckpt_name)
    if not os.path.exists(resume_path):
        return None

    ckpt = torch.load(resume_path, map_location=device)
    # 完整 checkpoint 的键：model/optimizer/scheduler/scaler/epoch/best_loss
    if isinstance(ckpt, dict) and "model" in ckpt:
        return {
            "format": "full",
            "model": ckpt["model"],
            "optimizer": ckpt.get("optimizer"),
            "scheduler": ckpt.get("scheduler"),
            "scaler": ckpt.get("scaler"),
            "start_epoch": int(ckpt.get("epoch", -1)) + 1,
            "best_loss": float(ckpt.get("best_loss", float("inf"))),
        }
    # 旧格式：纯 model state_dict（dict[str, Tensor]）
    return {
        "format": "old",
        "model": ckpt,
        "optimizer": None,
        "scheduler": None,
        "scaler": None,
        "start_epoch": 0,
        "best_loss": float("inf"),
    }


def build_training_components(device):
    """构建训练所需组件：分词器、模型、优化器、Noam 调度器、混合精度缩放器"""
    tokenizer = ZHTokenizer.from_vocab(config.ZH_VOCAB_PATH)
    model = InputMethod(
        zh_vocab_size=tokenizer.vocab_size,
        d_model=config.EMBEDDING_DIM,
        zh_pad_id=tokenizer.pad_id
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    # Noam 调度：先线性 warmup 上升，再按 1/sqrt(step) 缓慢衰减。
    # 相比 ExponentialLR 每 epoch 乘 0.94（后期学习率太小导致 loss 停滞），
    # 后期仍有足够学习能力，更适合大模型长时间训练
    warmup_steps = config.WARMUP_STEPS
    def lr_lambda(step):
        step = max(step, 1)
        return min(step / warmup_steps, 1.0) * (warmup_steps ** 0.5) * (step ** -0.5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    # 混合精度：T4 支持 fp16 tensor core，前向/反向减半显存并提速；
    # CPU 下 enabled=False 自动退化为 fp32，GradScaler 原样通过
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    return tokenizer, model, optimizer, scheduler, scaler


def train(rank, world_size):
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    # 诊断显存（Kaggle 排查 OOM 用）：打印 GPU 型号、总显存、占用
    if torch.cuda.is_available() and rank == 0:
        print(f"[GPU] {torch.cuda.get_device_name(rank)} | 总显存 {torch.cuda.get_device_properties(rank).total_memory / 2**30:.1f} GiB | PyTorch 已用 {torch.cuda.memory_allocated(rank) / 2**30:.2f} GiB")
    # DDP 仅在多进程且 CUDA 可用时启用；CPU 单进程直接跑裸模型
    use_ddp = torch.cuda.is_available() and world_size > 1
    # 构建模型/分词器/优化器/调度器/缩放器
    zhTokenizer, model, optimizer, scheduler, scaler = build_training_components(device)

    # 断点续训：读取 checkpoint 权重（必须在 DDP 包装之前加载，
    # 否则 state_dict 会带 module. 前缀导致键不匹配），恢复训练状态
    start_epoch = 0
    loss_value = float("inf")
    ckpt_data = load_checkpoint(device=device, ckpt_name="best_model.pth")
    if ckpt_data is not None:
        model.load_state_dict(ckpt_data["model"])
        if ckpt_data["format"] == "full" and ckpt_data["optimizer"] is not None:
            optimizer.load_state_dict(ckpt_data["optimizer"])
            scheduler.load_state_dict(ckpt_data["scheduler"])
            if ckpt_data["scaler"] is not None:
                scaler.load_state_dict(ckpt_data["scaler"])
            start_epoch = ckpt_data["start_epoch"]
            loss_value = ckpt_data["best_loss"]
            if rank == 0:
                print(f"[RESUME] 已恢复权重+优化器+调度器，从 Epoch {start_epoch+1} 继续（历史最佳 loss {loss_value:.4f}）")
        elif rank == 0:
            print("[RESUME] 检测到旧格式权重，已加载模型（LR 将重新走 warmup）")
    elif rank == 0:
        print("[RESUME] 未找到已有权重，从头开始训练")

    dataloader = get_dataloader(
        batch_size=config.BATCH_SIZE, shuffle=True, is_train=True,
        rank=rank if use_ddp else None,
        world_size=world_size if use_ddp else None
    )
    criterion = nn.CrossEntropyLoss(ignore_index=zhTokenizer.pad_id)
    if use_ddp:
        model = DDP(model, device_ids=[rank])

    # 只在 rank=0 创建 TensorBoard writer，避免两个进程重复写日志目录
    writer = SummaryWriter(log_dir=str(config.OUTPUT_DIR / "logs" / datetime.now().strftime('%Y%m%d_%H%M%S'))) if rank == 0 else None
    import time
    epoch_times = []

    for epoch in range(start_epoch, config.EPOCHS):
        # 每个 epoch 重新打乱数据（仅 DDP 的 DistributedSampler 需要 set_epoch）
        if hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch)
        t0 = time.time()
        loss_batch = train_one_epoch(model, dataloader, optimizer, criterion, device, scheduler, scaler)
        elapsed = time.time() - t0
        epoch_times.append(elapsed)
        if rank == 0:
            print(f"Epoch {epoch+1}, Loss: {loss_batch}, 耗时 {elapsed/60:.1f} 分钟")
            # 记录每个 epoch 的平均损失和学习率
            writer.add_scalar("loss/train", loss_batch, epoch)
            writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

        if loss_value > loss_batch:
            loss_value = loss_batch
            if rank == 0:
                # DDP 包装后取内部模型保存，避免保存带 module. 前缀的状态字典；
                # 连同 optimizer/scheduler/scaler 一起存，便于断点续训时 LR 曲线无缝衔接
                state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                ckpt = {
                    "model": state_dict,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "best_loss": loss_value,
                }
                torch.save(ckpt, os.path.join(config.CHECKPOINT_DIR, "best_model.pth"))

    if rank == 0 and epoch_times:
        avg_min = sum(epoch_times) / len(epoch_times) / 60
        print(f"训练完成，共 {len(epoch_times)} 轮，平均每轮 {avg_min:.1f} 分钟")

    if writer is not None:
        writer.close()

def train_one_epoch(model, dataloader, optimizer, criterion, device, scheduler=None, scaler=None):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for step, (batch, target) in enumerate(dataloader):
        batch = batch.to(device)
        target = target.to(device)
        # 混合精度前向：CUDA 下 fp16 加速，CPU 下自动 fp32
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            # output shape: (batch_size, vocab_size)，已是最后一位 token 的 logits
            output = model(batch)
            loss = criterion(output, target)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # Noam 调度按 batch 步数更新
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1

    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"[MEM] 本 epoch 峰值显存 {peak:.2f} GiB")
    return total_loss / num_batches if num_batches > 0 else float('inf')

def predict(input_str, model, tokenizer, device):
    ids = tokenizer.encode(input_str)
    if not ids:
        print(f"输入为空: {input_str}")
        return []
    ids = ids[-config.MAX_SEQ_LEN:]
    input_batch = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    topk_values, topk_indices = predict_batch(input_batch, model, device)

    indices = topk_indices[0].tolist()
    # print(f"输入: {input_str}")
    words = [tokenizer.id2word[idx] for idx in indices]
    # for idx, word in zip(indices, words):
    #     print(f"  {word} ({idx})")
    return words

def predict_batch(input_batch, model, device):
    model.eval()
    
    input_batch = input_batch.to(device)
    with torch.no_grad():
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            output = model(input_batch)
        topk_values, topk_indices = torch.topk(output, k=5, dim=1)
    return topk_values, topk_indices

def run():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt_data = load_checkpoint(device=device)
    if ckpt_data is None:
        print("[PREDICT] 未找到 best_model.pth，请先训练！")
        exit(1)
    tokenizer = ZHTokenizer.from_vocab(config.ZH_VOCAB_PATH)
    model = InputMethod(
        zh_vocab_size=tokenizer.vocab_size,
        d_model=config.EMBEDDING_DIM,
        zh_pad_id=tokenizer.pad_id
    ).to(device)
    model.load_state_dict(ckpt_data["model"])

    # 循环输入：每次输入一句中文，预测下一个词的 top-5 候选
    print("输入法预测已就绪，输入中文后回车查看候选词（输入 q / quit / exit 退出）")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not text:
            continue
        if text.lower() in ("q", "quit", "exit"):
            break
        words = predict(text, model, tokenizer, device)
        print(words)

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
    # if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
    #     world_size = torch.cuda.device_count()
    #     print(f"使用 {world_size} 张 GPU 进行 DDP 训练")
    #     mp.spawn(init_process, args=(world_size, train), nprocs=world_size)
    # else:
    #     print("未检测到多卡 GPU，单进程 CPU 训练")
    #     train(rank=0, world_size=1)
    run()
    
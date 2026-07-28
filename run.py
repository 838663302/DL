import torch
import os
import config
from model import MyModel
from dataset import getLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tokenizer import JiebaTokenizer

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"训练设备: {device}")

def read_json():
    with open(config.DATASET_DIR / "vocab.txt", "r", encoding="utf-8") as f:
        vocab_list = [vocab.strip() for vocab in f.readlines()]
    word2id = {word: i for i, word in enumerate(vocab_list)}
    id2word = {i: word for i, word in enumerate(vocab_list)}
    return word2id, id2word

def train(word2id):
    model = MyModel(len(word2id)).to(device)
    # 双卡并行：每个batch自动对半切分到两块GPU计算（数据常驻GPU0，切分为卡间P2P拷贝，开销小）
    use_dp = torch.cuda.device_count() > 1
    if use_dp:
        print(f"检测到 {torch.cuda.device_count()} 块GPU，启用 DataParallel 并行训练")
        model = torch.nn.DataParallel(model)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    model.train()
    loss_value = float("inf")
    # 数据直接常驻GPU，训练循环中无CPU取数与拷贝开销
    train_loader = getLoader(True, device=device)

    # 确保模型保存目录存在
    os.makedirs(config.MODEL_PATH.parent, exist_ok=True)
    writer = SummaryWriter(log_dir=str(config.OUTPUT_DIR / "logs" / datetime.now().strftime('%Y%m%d_%H%M%S')))
    
    for epoch in range(config.EPOCHS):
        # loss在GPU上累加，每轮只在结尾.item()同步一次，避免每步强制CPU/GPU同步
        loss_sum = torch.zeros(1, device=device)
        num_batches = 0
        for batch, target in train_loader:
            optimizer.zero_grad()
            output = model(batch)
            # output: (batch_size, vocab_size), target: (batch_size)
            loss = loss_func(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_sum += loss.detach()
            num_batches += 1
        avg_loss = (loss_sum / num_batches).item() if num_batches else 0.0
        scheduler.step()
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        print(f"Epoch {epoch+1}/{config.EPOCHS} | avg_loss: {avg_loss:.4f} | lr: {optimizer.param_groups[0]['lr']:.6f}")
        if avg_loss < loss_value:
            loss_value = avg_loss
            # DataParallel包装后真实模型在model.module中，保存它以保证单卡也能加载
            raw_model = model.module if use_dp else model
            torch.save(raw_model.state_dict(), config.MODEL_PATH)
    writer.close()

def predict(input_str, model):
    tokenizer = JiebaTokenizer.from_vocab(config.DATASET_DIR / "vocab.txt")
    input_batch = tokenizer.encode(input_str)
    input_batch = torch.tensor(input_batch, dtype=torch.long).unsqueeze(0)
    topk_values, topk_indices = predict_batch(input_batch, model)

    indices = topk_indices[0].tolist()
    print(f"输入: {input_str}")
    words = [tokenizer.id2word[idx] for idx in indices]
    for idx, word in zip(indices, words):
        print(f"  {word} ({idx})")
    return words

def predict_batch(input_batch, model):
    model.eval()
    with torch.no_grad():
        output = model(input_batch.to(device))
        topk_values, topk_indices = torch.topk(output, k=5, dim=1)
    return topk_values, topk_indices

def evalute(model):
    test_loader = getLoader(False, device=device)
    top_value = 0
    topk_value = 0
    total = 0
    for batch, targets in test_loader:
        _, topk_indices = predict_batch(batch, model)
        targets = targets.to(topk_indices.device)  
        total += targets.size(0)

        # top-1：第 0 列 == target
        top_value  += (topk_indices[:, 0] == targets).sum().item()
        # top-5：广播比较 (B,5) vs (B,1)，每行是否命中 target
        topk_value += (topk_indices == targets.unsqueeze(1)).any(dim=1).sum().item()
    print(f"Top-1 Accuracy: {top_value / total:.4f}")
    print(f"Top-5 Accuracy: {topk_value / total:.4f}")

    
if __name__ == "__main__":
    word2id, id2word = read_json()
    # model = MyModel(len(word2id)).to(device)
    # model.load_state_dict(torch.load(config.MODEL_PATH))
    
    train(word2id)

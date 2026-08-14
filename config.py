from pathlib import Path

# Kaggle环境检测：输入数据在/kaggle/input（只读），输出必须写到/kaggle/working
if Path('/kaggle/input').exists():
    # 数据集实际挂载路径（线上实测）
    BASE_DIR = Path('/kaggle/input/datasets/xiaonanhaiaichixigua/inputmethoddata')
    OUTPUT_DIR = Path('/kaggle/working')
else:
    BASE_DIR = Path(__file__).parent.resolve()
    OUTPUT_DIR = BASE_DIR

DATASET_DIR = BASE_DIR / "data"
WINDOW_SIZE = 5
# 输入法模型极小（窗口仅 8 token），显存余量巨大，但 200 万条滑窗样本高度重叠，
# 大 batch 的梯度优势不明显，反而会让后期 LR 衰减过慢、训练不稳（epoch 16 后 loss 反弹）。
# 故回退 batch 以增加每 epoch 步数，配合小 LR 更稳。
# DDP 下全局有效 batch = 1024 * 2 = 2048
BATCH_SIZE = 1024
EMBEDDING_DIM = 256
HIDDEN_SIZE = 256
# batch 减半后按线性缩放规则 LR 同步减半（Noam 峰值）。
# 实测 0.004 在 epoch 16 后 loss 反弹卡死，逐级回退到 0.001
LR = 0.001  # Noam 调度的峰值学习率（warmup 结束后达到）
# 200 万条滑窗样本高度重叠，50 轮过拟合；15 轮足够收敛
EPOCHS = 40
WARMUP_STEPS = 2000  # Noam 调度的 warmup 步数，之后学习率按 1/sqrt(step) 缓慢衰减
MODEL_PATH = OUTPUT_DIR / "model.pth"
MAX_SEQ_LEN = 128  # 最大序列长度

# 束搜索解码参数
BEAM_SIZE = 5              # 束宽：每步保留的候选数，越大越慢但质量更高
LENGTH_PENALTY_ALPHA = 0.6 # Google NMT 长度惩罚系数，>0 鼓励长句

# 词汇表路径
ZH_VOCAB_PATH = DATASET_DIR / "iwslt_train_zh_vocab.txt"
EN_VOCAB_PATH = DATASET_DIR / "iwslt_train_en_vocab.txt"

# 检查点目录：直接使用输出根目录，不再单独建 checkpoints 子目录
CHECKPOINT_DIR = OUTPUT_DIR
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Transformer模型参数
NHEAD = 8  # 多头注意力头数（d_model=256 时每头 32 维，必须整除 d_model）
NUM_ENCODER_LAYERS = 3  # 编码器层数
NUM_DECODER_LAYERS = 3  # 解码器层数
DIM_FEEDFORWARD = 1024  # 前馈网络维度（取 4*d_model）
DROPOUT = 0.1  # Dropout比率
ACTIVATION = 'relu'  # 激活函数
INPUY_WINDOW_SIZE = 8

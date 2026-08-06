from pathlib import Path

# Kaggle环境检测：输入数据在/kaggle/input（只读），输出必须写到/kaggle/working
if Path('/kaggle/input').exists():
    # 数据集实际挂载路径（线上实测）
    BASE_DIR = Path('/kaggle/input/datasets/xiaonanhaiaichixigua/reviewdata')
    OUTPUT_DIR = Path('/kaggle/working')
else:
    BASE_DIR = Path(__file__).parent.resolve()
    OUTPUT_DIR = BASE_DIR

DATASET_DIR = BASE_DIR / "data"
WINDOW_SIZE = 5
# DDP 下每张卡独立跑 batch=64，全局有效 batch = 64 * 2 = 128
# 每卡显存压力与单卡相同，无需降低
BATCH_SIZE = 64
EMBEDDING_DIM = 128
HIDDEN_SIZE = 256
LR = 0.001  # 等效batch回落，学习率同步回调
EPOCHS = 30
MODEL_PATH = OUTPUT_DIR / "model.pth"
MAX_SEQ_LEN = 128  # 最大序列长度

# 词汇表路径
ZH_VOCAB_PATH = DATASET_DIR / "iwslt_train_zh_vocab.txt"
EN_VOCAB_PATH = DATASET_DIR / "iwslt_train_en_vocab.txt"

# 检查点目录：直接使用输出根目录，不再单独建 checkpoints 子目录
CHECKPOINT_DIR = OUTPUT_DIR
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Transformer模型参数
NHEAD = 4  # 多头注意力头数（d_model=128 时每头 32 维）
NUM_ENCODER_LAYERS = 3  # 编码器层数
NUM_DECODER_LAYERS = 3  # 解码器层数
DIM_FEEDFORWARD = 256  # 前馈网络维度（缩小版取 2*d_model）
DROPOUT = 0.1  # Dropout比率
ACTIVATION = 'relu'  # 激活函数

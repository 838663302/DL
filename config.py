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
BATCH_SIZE = 64  # Transformer的logits是(batch×seq×vocab)量级，512过大导致OOM，调小为64
EMBEDDING_DIM = 128
HIDDEN_SIZE = 256
LR = 0.001  # batch调小后学习率同步下调，保持收敛稳定
EPOCHS = 50
MODEL_PATH = OUTPUT_DIR / "model.pth"
MAX_SEQ_LEN = 128  # 最大序列长度

# 词汇表路径
ZH_VOCAB_PATH = DATASET_DIR / "iwslt_train_zh_vocab.txt"
EN_VOCAB_PATH = DATASET_DIR / "iwslt_train_en_vocab.txt"

# 检查点目录
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Transformer模型参数
NHEAD = 8  # 多头注意力头数
NUM_ENCODER_LAYERS = 6  # 编码器层数
NUM_DECODER_LAYERS = 6  # 解码器层数
DIM_FEEDFORWARD = 512  # 前馈网络维度（通常为d_model*4）
DROPOUT = 0.1  # Dropout比率
ACTIVATION = 'relu'  # 激活函数

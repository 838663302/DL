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
BATCH_SIZE = 512  # 双卡并行时每卡分到一半，调大以提高GPU利用率并减少每epoch的迭代次数
EMBEDDING_DIM = 128
HIDDEN_SIZE = 256
LR = 0.002  # 随batch调大同步上调
EPOCHS = 50
MODEL_PATH = OUTPUT_DIR / "model.pth"

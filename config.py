import os
from pathlib import Path

# Kaggle环境检测：输入数据在/kaggle/input（只读），输出必须写到/kaggle/working
if Path('/kaggle/input').exists():
    BASE_DIR = Path(os.environ.get('KAGGLE_INPUT', '/kaggle/input'))
    OUTPUT_DIR = Path('/kaggle/working')
else:
    BASE_DIR = Path(__file__).parent.resolve()
    OUTPUT_DIR = BASE_DIR

DATASET_DIR = BASE_DIR / "data"
WINDOW_SIZE = 5
BATCH_SIZE = 42
EMBEDDING_DIM = 128
HIDDEN_SIZE = 256
LR = 0.001
EPOCHS = 50
MODEL_PATH = OUTPUT_DIR / "model.pth"
